"""Join frozen-state mesh geometry for two raw action blocks to real execution."""
import argparse
from collections import defaultdict
import csv
import gzip
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from archive import Trial, digest, read
from g2_assets import load_arms
from g2_kinematics import pose_error
from g2_robotwin_backend import pose7
from robotwin_control_geometry import inverse_pose, pose_from_wxyz
from verify_robotwin_fk import Chain


def trace(events, start_by_query, *, archived_tail=False):
    rows = defaultdict(list)
    query = 2 if archived_tail else 0
    row = 0
    for line, event in events:
        if archived_tail:
            row = event["original_policy_row"]
        elif event["kind"] == "policy_request":
            query += 1
            row = 0
        elif event["kind"] == "approved_plan":
            row = event["row_in_prefix"]
        if query in start_by_query and row and event.get("physics_step_count"):
            state = event["contact_state"]
            if state:
                delta = np.array(state["block_pose_task"])[:3,3] - start_by_query[query]
                rows[query,row].append((line,event,1000*float(np.linalg.norm(delta[:2]))))
    return rows


def physical_metrics(found):
    if not found:
        return {}
    contact = [(line,e) for line,e,_ in found if e["contact_state"]["contact_fingers"]["left"]]
    return dict(step_count=len(found), tick_start=found[0][1]["tick"],tick_end=found[-1][1]["tick"],
                first_left_contact_tick=contact[0][1]["tick"] if contact else None,
                first_left_contact_line=contact[0][0] if contact else None,
                bilateral_left_steps=sum(len(e["contact_state"]["contact_fingers"]["left"])==2 for _,e,_ in found),
                left_contact_steps=len(contact),
                stable_left_steps=sum("left" in e["contact_state"]["holders"] for _,e,_ in found),
                measured_left_grip_end=float(np.mean(found[-1][1]["actual_fingers"][0])/.044),
                max_block_xy_mm=max(d for _,_,d in found),
                block_xy_end_mm=found[-1][2])


def load_geometry(trial, meshes, assets, audit, existing):
    raw_events=list(trial.events())
    requests=[e for _,e in raw_events if e["kind"]=="policy_request"][:2]
    cite_path=existing/"dynamic/q2-final-tail/events.jsonl.gz"
    cpu_raw=gzip.decompress(cite_path.read_bytes())
    prior=read(existing/"dynamic/physical-summary.json")["q2-final-tail"]
    assert digest(cpu_raw)==prior["full_events_sha256"]
    gpu=trace(raw_events,{i:np.array(e["objects_task"]["block"])[:3,3]
                          for i,e in enumerate(requests,1)})
    original=read(existing/"inputs/q2-block8.json")
    cpu=trace(((i,json.loads(line)) for i,line in enumerate(cpu_raw.splitlines(),1)),
              {2:np.array(original["at"]["contact_state"]["block_pose_task"])[:3,3]},
              archived_tail=True)
    robots = ("aloha-agilex","ARX-X5")
    arms = [load_arms(assets,audit,name,trial.config["calibration"][name]) for name in robots]
    urdf={e["robot"]:assets/e["urdf_member"] for e in read(audit)["robots"]}
    fingers=[[[Chain(urdf[name],joint) for joint in arm.grip.names] for arm in pair]
             for name,pair in zip(robots,arms)]
    geometry=[]
    for q, request in enumerate(requests,1):
        action_bytes=trial.verified(trial.episode/f"action-{q:05}.npy")
        actions=np.load(BytesIO(action_bytes),allow_pickle=False)
        receipt=read(meshes/f"mesh-q{q}/receipt.json")
        assert receipt["status"]=="complete" and receipt["source_action_sha256"]==digest(action_bytes)
        assert len(receipt["records"])==50
        world_task=np.asarray(request["world_task"])
        block=np.asarray(request["objects_task"]["block"])
        source_world=pose_from_wxyz(trial.config["source_world_task"][:3],trial.config["source_world_task"][3:])
        source=[inverse_pose(source_world)@pose_from_wxyz(p[:3],p[3:]) for p in trial.config["source_base_world"]]
        target=[inverse_pose(world_task)@np.asarray(e["world_base"]) for e in request["actual"]]
        for index,(raw,mesh) in enumerate(zip(actions,receipt["records"],strict=True),1):
            assert mesh["row"]==index
            np.testing.assert_array_equal(mesh["original_action"],raw)
            data=dict(query=q,row=index,query_tick=request["stamp"]["tick"],
                      action_sha256=digest(action_bytes),archive_raw_row=raw.tolist(),
                      query_block_pose_task=pose7(block),frozen_ARX_production_ik_clear=mesh["production_ik_clear"])
            values=raw.reshape(2,7)
            source_tools=np.asarray(mesh["original_tools_task"])
            target_q=np.asarray(mesh["target_q"])
            for side,i in (("left",0),("right",1)):
                grasp=float(values[i,6])
                data[f"{side}_raw_grip"]=grasp
                for name,group,frames,qpos in (
                    ("aloha",0,source,values[:,:6]),("arx",1,target,target_q)):
                    arm=arms[group][i]
                    tool=frames[i]@arm.chain.fk(qpos[i])@arm.offset()
                    if group==0:
                        np.testing.assert_allclose(tool,source_tools[i],atol=1e-9,rtol=0)
                    residual=pose_error(source_tools[i],tool)
                    if group==1:
                        data[f"{side}_arx_ik_pos_error_mm"]=1000*np.linalg.norm(residual[:3])
                        data[f"{side}_arx_ik_angle_error_deg"]=np.rad2deg(np.linalg.norm(residual[3:]))
                    prefix=f"{side}_{name}"
                    native=arm.grip.target(float(np.clip(grasp,0,1)))
                    data[f"{prefix}_tcp_task_xyz_wxyz"]=pose7(tool)
                    data[f"{prefix}_block_in_tcp_xyz_wxyz"]=pose7(inverse_pose(tool)@block)
                    data[f"{prefix}_finger_joint_request_m"]=native.tolist()
                    values_by_name=dict(zip(arm.chain.names+arm.grip.names,np.r_[qpos[i],native]))
                    for j,chain in enumerate(fingers[group][i],1):
                        fp=frames[i]@chain.fk(values_by_name)
                        data[f"{prefix}_finger{j}_link_origin_task_m"]=fp[:3,3].tolist()
                    mesh_key="source" if group==0 else "target"
                    data[f"{prefix}_depth_start_mm"]=mesh[f"{mesh_key}_start_finger_depth_mm"][i]
                    data[f"{prefix}_depth_requested_mm"]=mesh[f"{mesh_key}_requested_finger_depth_mm"][i]
                    data[f"{prefix}_max_depth_start_mm"]=max(data[f"{prefix}_depth_start_mm"])
                    data[f"{prefix}_max_depth_requested_mm"]=max(data[f"{prefix}_depth_requested_mm"])
                    data[f"{prefix}_block_y_mm"]=1000*data[f"{prefix}_block_in_tcp_xyz_wxyz"][1]
            data.update(gpu_actual="MEASURED" if gpu[q,index] else "NOT_EXECUTED",
                        cpu_tail="MEASURED" if cpu[q,index] else "NOT_REPLAYED")
            geometry.append((data,physical_metrics(gpu[q,index]),physical_metrics(cpu[q,index])))
    return geometry


def write_csv(rows, path):
    mixed=[]
    for data, gpu, cpu in rows:
        entry=dict(data)
        entry.update({f"gpu_{k}":v for k,v in gpu.items()})
        entry.update({f"cpu_tail_{k}":v for k,v in cpu.items()})
        mixed.append({k:json.dumps(v,separators=(",",":")) if isinstance(v,list) else v for k,v in entry.items()})
    keys=list(dict.fromkeys(k for row in mixed for k in row))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,lineterminator="\n")
        writer.writeheader()
        writer.writerows(mixed)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","meshes","assets","asset_audit","prior_physics","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    write_csv(load_geometry(Trial(args.archive,"127999421"),args.meshes,args.assets,args.asset_audit,
                            args.prior_physics),args.output)
