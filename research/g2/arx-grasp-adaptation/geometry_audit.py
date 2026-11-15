"""Compare mesh pad frames and measured first grasp for Aloha and ARX archives."""
import argparse
from collections import deque
from hashlib import file_digest
import gzip
import json
from pathlib import Path

import numpy as np

from g2_arx_grasp_adaptation import load_jaws
from g2_assets import load_arms
from g2_initial import matrix7
from g2_kinematics import pose_error
from robotwin_control_geometry import inverse_pose


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def pose_geometry(tool, block, fingers, arm, geom):
    fractions = (np.asarray(fingers)-arm.grip.closed)/geom.travel_m
    inner_faces = geom.faces(fractions)
    relative = inverse_pose(tool) @ block
    centers = inner_faces.mean(axis=1)
    def shifted(v):
        frame = np.eye(4)
        frame[:3,3] = -v
        return frame @ relative
    return dict(block_in_tcp=relative,
                block_in_grasp_center=shifted(centers.mean(axis=0)),
                block_in_finger_face_frames=[shifted(v) for v in centers],
                face_centers_in_block=[(inverse_pose(block) @ tool @ np.r_[v,1])[:3] for v in centers],
                raw_finger_q_m=fingers, calibrated_opening=arm.grip.read(fingers),
                tcp_xyz=tool[:3,3])


def aloha_grasps(root, arm, geom):
    manifest=json.loads((root/"RAW-SHA256.json").read_text())
    summaries={}
    for job in ("127589584","127589587","127589590"):
        f,=root.glob(f"results/pace-g2-parallel/*/{job}/trials/*/physics.jsonl")
        assert sha(f)==manifest[str(f.relative_to(root))]
        recent=deque(maxlen=26)
        stages={}
        for lineno,line in enumerate(f.open(),1):
            e=json.loads(line)
            if lineno==1:
                names=e["joint_names"]
                initial=np.asarray(e["block_pose"])
            q=np.asarray(e["qpos"])
            tool=np.asarray(e["robot_root"]) @ arm.chain.fk(q[[names.index(n) for n in arm.chain.names]]) @ arm.offset()
            recent.append(tool[:3,3])
            block=np.asarray(e["block_pose"])
            matching={}
            for record in e["contacts"]:
                pair=set(record["pair"])
                if "box" not in pair:
                    continue
                for name in pair & {"fl_link7","fl_link8"}:
                    points=[p["position"] for p in record["points"] if np.linalg.norm(p["impulse"])>=1e-8]
                    if points:
                        matching[name]=[(inverse_pose(block)@np.r_[point,1])[:3] for point in points]
            cmd=e["normalized_gripper_command"][0]
            status=dict(first_impulse=bool(matching),first_bilateral=len(matching)==2,
                        first_req_le_095=cmd<=.95,first_req_le_05=cmd<=.5)
            for name,found in status.items():
                if found and name not in stages:
                    canonical=matrix7(e["tcp"][0]) @ np.diag([1,-1,-1,1])
                    calibration_error=pose_error(canonical,tool)
                    stages[name]=dict(line=lineno,tick=e["tick"],row=e["action"],left_request=cmd,
                        native_tcp_to_canonical_error=calibration_error,
                        approach_last_100ms_task_m=recent[-1]-recent[0],
                        approach_last_100ms_block_m=block[:3,:3].T @ (recent[-1]-recent[0]),
                        block_planar_shift_m=np.linalg.norm((block-initial)[:2,3]),
                        block_contact_points_m=matching,
                        **pose_geometry(tool,block,q[[names.index(n) for n in arm.grip.names]],arm,geom))
            if len(stages)==4:
                break
        summaries[job]=dict(physics_sha256=sha(f),initial_block_xyz=initial[:3,3],stages=stages)
    return summaries


def arx_grasps(root, outcomes, arm, geom):
    study=Path(__file__).resolve().parents[1]/"arx-online-batch-20261115"
    result={}
    analysis=json.loads(gzip.decompress((study/"root-cause/action-geometry.json.gz").read_bytes()))
    for label,run in outcomes.items():
        if label not in {"G1","G2","G3"}:
            continue
        data=root/run["result_directory"]/"trials"/run["episode"]/"episode"
        report=analysis[label]
        relevant={(1,report["rows"][0]["row"])}
        close=report["query_stats"][1]["first_closing_rows"]["0.5"]
        target=next(r for r in report["rows"] if (r["query"],r["row"])==(2,close))
        relevant.add((2,close))
        last_executed=max(report["query_stats"][1]["executed_rows"])
        relevant.add((2,last_executed))
        stages={}
        query=row=0
        first=False
        recent=deque(maxlen=26)
        for lineno,line in enumerate((data/"events.jsonl").open(),1):
            e=json.loads(line)
            if e["kind"]=="policy_request":
                query+=1
                continue
            if query>2:
                break
            if e["kind"]=="direct_row_planning":
                row=e["row_in_prefix"]
                continue
            if not e.get("physics_step_count"):
                continue
            contact=e["contact_state"]
            tool=np.asarray(e["actual_tools_task"][0])
            block=np.asarray(contact["block_pose_task"])
            recent.append(tool[:3,3])
            keys=[]
            if query==1 and not first and contact["contact_fingers"]["left"]:
                keys.append("first_impulse")
                first=True
            if (query,row) in relevant:
                keys.append(f"q{query}_last_step_row{row}")
            for key in keys:
                stages[key]=dict(query=query,row=row,line=lineno,tick=e["tick"],
                    approach_last_100ms_block_m=block[:3,:3].T @ (recent[-1]-recent[0]),
                    physical_contacts=e.get("physical_contacts"),contact_fingers=contact["contact_fingers"],
                    **pose_geometry(tool,block,np.asarray(e["actual_fingers"][0]),arm,geom))
        raw=np.array(target["block_at_query_in_target_tcp_mm"][0])/1000
        positions=np.asarray(target["grip_request"][0]).clip(0,1)
        result[label]=dict(job=run["job_id"],closure_row=close,row_executed=close in report["query_stats"][1]["executed_rows"],
                    predicted_block_tcp_m=raw,predicted_block_in_center_m=raw-geom.center([positions]*2),
                    stages=stages)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("input","audit","assets","source_raw","target_raw","output"):
        p.add_argument(n,type=Path)
    a=p.parse_args()
    cfg=json.loads(a.input.read_text())["config"]
    members={r["robot"]:r["urdf_member"] for r in json.loads(a.audit.read_text())["robots"]}
    robots=("aloha-agilex","ARX-X5")
    arms=[load_arms(a.assets,a.audit,n,cfg["calibration"][n])[0] for n in robots]
    geoms=[load_jaws(a.assets/members[n],arm) for arm,n in zip(arms,robots)]
    batch=Path(__file__).resolve().parents[1]/"arx-online-batch-20261115"
    output=dict(jaws={n:dict(tcp_offset=arm.offset(),finger_travel_m=g.travel_m,
        meshes_sha256=g.meshes_sha256,closed_face_extents_tcp_m=g.faces([0,0]),
        opened_face_extents_tcp_m=g.faces([1,1]),closed_center_in_tcp_m=g.center(),
        opened_center_in_tcp_m=g.center([1,1])) for n,arm,g in zip(robots,arms,geoms)},
        native_aloha=aloha_grasps(a.source_raw,arms[0],geoms[0]),
        arx=arx_grasps(a.target_raw,json.loads((batch/"submission_receipts.json").read_text())["candidates"],arms[1],geoms[1]))
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False,default=lambda v:v.tolist())+"\n")
