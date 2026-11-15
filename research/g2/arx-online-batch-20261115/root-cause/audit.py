"""Trace archived plans and PhysX against the frozen source FK and target IK."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import tarfile

import numpy as np

from g2_assets import load_arms
from g2_initial import matrix7
from g2_kinematics import pose_error
from robotwin_control_geometry import inverse_pose

STUDY = Path(__file__).resolve().parents[1]
CANDIDATES = STUDY.parent / "arx-stage6-concentrated-20261114/gpu-candidates"


def read(path):
    return json.loads(path.read_text())


def configs(label, episode):
    parent = CANDIDATES if label in {"G1", "G2"} else STUDY / "packages"
    with tarfile.open(parent / label / "g2-standalone-arx.tar.gz") as archive:
        return json.load(archive.extractfile(f"g2-standalone-arx/configs/{episode}.json"))


def fk(arms, frames, q):
    return np.array([frame @ arm.chain.fk(values) @ arm.offset()
                     for arm, frame, values in zip(arms, frames, q)])


def errors(goal, actual):
    delta = np.asarray([pose_error(a, b) for a, b in zip(np.asarray(goal), np.asarray(actual))])
    return dict(mm=1000 * np.linalg.norm(delta[:, :3], axis=1),
                deg=np.rad2deg(np.linalg.norm(delta[:, 3:], axis=1)))


def offsets(tools, block):
    return 1000 * np.array([(inverse_pose(tool) @ block)[:3, 3] for tool in tools])


def extent(points):
    return dict(min=np.min(points, axis=0), max=np.max(points, axis=0)) if len(points) else None


def audit_trial(label, spec, root, assets, audit, snapshots):
    run = root / spec["result_directory"]
    trial = run / "trials" / spec["episode"]
    data = trial / "episode"
    result = read(trial / "receipt.json")["result"]
    cfg = configs(label, spec["episode"])
    start = result["initial"]
    source, target = [load_arms(assets, audit, name, cfg["calibration"][name])
                      for name in ("aloha-agilex", "ARX-X5")]
    sframe = [inverse_pose(matrix7(cfg["source_world_task"])) @ matrix7(v) for v in cfg["source_base_world"]]
    tframe = [inverse_pose(matrix7(cfg["world_task"])) @ matrix7(start["robot_base_world"][side])
              for side in ("left", "right")]
    target_open = np.array([arm.grip.target(1.) for arm in target])
    target_closed = np.array([arm.grip.target(0.) for arm in target])

    def grip(fingers):
        return np.mean((np.asarray(fingers) - target_closed) / (target_open - target_closed), axis=1)

    episode_start = matrix7(start["objects_task"]["block"])
    request, track = {}, defaultdict(list)
    approvals = defaultdict(list)
    executed = defaultdict(list)
    event_lines = {}
    event_digest = sha256()
    q = row = 0
    previous_step = None
    last_plan = None
    with (data / "events.jsonl").open("rb") as stream:
        for number, line in enumerate(stream, 1):
            event_digest.update(line)
            e = json.loads(line)
            kind = e["kind"]
            if kind == "policy_request":
                q += 1
                request[q] = e
                event_lines[q] = number
            elif kind == "direct_row_planning":
                row = e["row_in_prefix"]
            elif kind == "approved_plan":
                approvals[q, row].append((number, e))
                last_plan = e
            elif e.get("physics_step_count"):
                executed[q, row].append((number, e))
                if "position_error_m" in e:
                    track[q].append([e["position_error_m"], e["orientation_error_deg"]])
                previous_step = e

    report = dict(job=spec["job_id"], scene_seed=cfg["scene_seed"], policy_seed=cfg["policy_rng_seed"],
                  events_sha256=event_digest.hexdigest(), query_stats=[], rows=[],
                  failure=dict(query=q, row=row, **result["failure"]))
    terminal = result["failure"]["diagnostic"]
    for index, record in request.items():
        action = data / f"action-{index:05}.npy"
        chunk = np.load(action, allow_pickle=False)
        assert chunk.shape == (50, 14) and np.isfinite(chunk).all()
        obs_block = np.asarray(record["objects_task"]["block"])
        assert np.max(np.abs(np.asarray(record["world_task"]) - matrix7(cfg["world_task"]))) < 1e-9
        mapped_state = np.asarray(record["state"]).reshape(2, 7)
        names = [sample["names"] for sample in record["actual"]]
        qpos = np.array([[sample["qpos"][order.index(name)] for name in arm.chain.names]
                         for sample, order, arm in zip(record["actual"], names, target)])
        tfb = fk(target, tframe, qpos)
        stat = dict(query=index, request_line=event_lines[index], tick=record["stamp"]["tick"],
                    action_sha256=sha256(action.read_bytes()).hexdigest(),
                    observation_fk_roundtrip=errors(tfb, fk(source, sframe, mapped_state[:, :6])),
                    block_xy_from_episode_start_mm=1000*np.linalg.norm((obs_block-episode_start)[:2,3]),
                    observed_block_xyz_mm=1000*obs_block[:3,3], observed_tcp_relative_block_mm=offsets(tfb, obs_block),
                    target_tracking=extent(np.asarray(track[index])) if track[index] else None)
        predicted = []
        measured = []
        for i, raw in enumerate(chunk, 1):
            values = raw.reshape(2, 7)
            goal = fk(source, sframe, values[:, :6])
            raw_relative = offsets(goal, obs_block)
            record_row = dict(query=index, row=i, grip_request=values[:, 6],
                              raw_action=raw, target_tcp_xyz_mm=1000*goal[:, :3,3],
                              target_delta_from_observation=errors(goal, tfb),
                              block_at_query_in_target_tcp_mm=raw_relative,
                              segments=len(approvals[index, i]))
            predicted.append(raw_relative)
            if approvals[index, i]:
                plans = approvals[index, i]
                stamp = plans[-1][1]["receipt"]["stamp"]["tick"]
                detail = plans[-1][1]["receipt"]
                replanned_goal = (detail.get("requests") or [detail.get("original_request")])[-1]
                record_row.update(last_plan_line=plans[-1][0], plan_tick=stamp,
                                  partial=bool(detail.get("partial_policy_row")))
                if replanned_goal is not None:
                    logged_goal = np.asarray(replanned_goal["task_tools"])
                    np.testing.assert_allclose(logged_goal, goal, atol=1e-9, rtol=0)
                    ik_goal = replanned_goal.get("ik_target", detail.get("original_ik_target"))
                    if ik_goal is not None:
                        record_row["mapping_fk_error"]=errors(goal, fk(target, tframe, ik_goal))
            if executed[index, i]:
                steps=executed[index, i]
                first,last = steps[0][1],steps[-1][1]
                xyz=np.array([e["contact_state"]["block_pose_task"] for _, e in steps])
                tcp=np.array([e["actual_tools_task"] for _,e in steps])
                current=np.array([offsets(t,b) for t,b in zip(tcp,xyz)])
                changes=1000*(xyz[:,:3,3]-obs_block[:3,3])
                measured.extend((e["tick"],grip(e["actual_fingers"]),e["contact_state"]["contact_fingers"])
                                for _,e in steps)
                record_row.update(first_step_line=steps[0][0],last_step_line=steps[-1][0],
                                  start_tick=first["tick"],end_tick=last["tick"],
                                  physical_steps=sum(e["physics_step_count"] for _,e in steps),
                                  measured_aperture=extent([grip(e["actual_fingers"]) for _,e in steps]),
                                  measured_tcp_grasp_error_end=errors(goal,tcp[-1]),
                                  live_block_in_actual_tcp_mm=extent(current),
                                  end_block_in_actual_tcp_mm=current[-1],
                                  live_block_in_policy_tcp_mm=offsets(goal,xyz[-1]),
                                  block_xy_from_query_mm=float(np.max(np.linalg.norm(changes[:,:2],axis=1))),
                                  max_fingers=[max(len(e["contact_state"]["contact_fingers"][side])
                                                   for _,e in steps) for side in ("left","right")])
            report["rows"].append(record_row)
        executed_numbers=[i for i in range(1,51) if executed[index,i]]
        left=chunk[:,6]
        stat.update(left_aperture_req_range=extent(left),
                    first_closing_rows={str(limit):next((i for i,x in enumerate(left,1) if x<=limit),None)
                                        for limit in (.95,.5,.1)},
                    executed_rows=executed_numbers,
                    executed_left_aperture_req_min=min((left[i-1] for i in executed_numbers),default=None),
                    actual_aperture=extent([x[1] for x in measured]),
                    left_contact_steps=sum(bool(x[2]["left"]) for x in measured),
                    query_block_in_policy_tcp_mm=extent(np.asarray(predicted)))
        report["query_stats"].append(stat)
    fail_goal=fk(source,sframe,np.load(data/f"action-{q:05}.npy",allow_pickle=False)[row-1].reshape(2,7)[:,:6])
    fail=report["failure"]
    for hit in terminal.get("requests",[]):
        np.testing.assert_allclose(hit["task_tools"],fail_goal,atol=1e-9,rtol=0)
        if "ik_target" in hit:
            fail["terminal_ik_residual"]=errors(fail_goal,fk(target,tframe,hit["ik_target"]))
    if previous_step is not None:
        end_block=previous_step["contact_state"]["block_pose_task"]
        fail["latest_block_in_policy_tcp_mm"]=offsets(fail_goal,end_block)
        fail["latest_measured_tcp_delta_from_goal"]=errors(fail_goal,previous_step["actual_tools_task"])
        fail["last_measured_geometry"] = previous_step.get("actual_mesh_collisions")
    if result["failure"]["phase"] == "control" and last_plan is not None:
        with np.load(data/last_plan["trajectory_file"],allow_pickle=False) as path:
            tick = previous_step["tick"]
            delta = tick-last_plan["receipt"]["stamp"]["tick"]
            substeps = round(float(path["period_s"])/previous_step["period_s"])
            index = delta//substeps + 1
            if delta % substeps == 0 and index < len(path["q"]):
                actual_tool = np.asarray(previous_step["actual_tools_task"])
                next_tool = fk(target,tframe,path["q"][index])
                fail["runtime_recheck"] = dict(tick=tick,control_index=index,
                    trajectory=last_plan["trajectory_file"],
                    next_tcp_minus_actual_task_xyz_mm=1000*(next_tool[:,:3,3]-actual_tool[:,:3,3]))
    fail["raw_tcp_xyz_mm"]=1000*fail_goal[:,:3,3]
    fail["candidate_counts"]=Counter()
    fail["candidate_depth_mm"]=defaultdict(list)
    fail["candidate_residuals"]=[]
    if "measured_start_q" in terminal:
        measured_start = np.asarray(terminal["measured_start_q"])
        fail["goal_delta_from_measured_start"]=errors(fail_goal,fk(target,tframe,measured_start))
    if terminal.get("candidate_verdicts"):
        ordered = sorted(terminal["candidate_verdicts"], key=lambda candidate:
                         np.linalg.norm(np.asarray(candidate["candidate_q"]) - measured_start))
        fail["nearest_branch_delta_rad"]=np.asarray(ordered[0]["candidate_q"])-measured_start
        fail["nearest_branch_collisions"]=ordered[0]["collisions"]
    for verdict in terminal.get("candidate_verdicts",[]):
        mapped=fk(target,tframe,verdict["candidate_q"])
        fail["candidate_residuals"].append(errors(fail_goal,mapped))
        for hit in verdict.get("collisions") or ():
            key=" / ".join(str(hit[k]) for k in ("object_name1","link_name1","object_name2","link_name2"))
            fail["candidate_counts"][key]+=1
            if hit.get("max_penetration_m") is not None:
                fail["candidate_depth_mm"][key].append(1000*hit["max_penetration_m"])
    fail["candidate_depth_mm"]={key:extent(dep) for key,dep in fail["candidate_depth_mm"].items()}
    if snapshots is not None:
        # Use the last real measured tick. A no-motion initial rejection uses actual reset joints.
        at=previous_step
        if at is None:
            ar=np.array([start["actual_arm_qpos_rad"][side] for side in ("left","right")])
            fg=np.array([[sample["qpos"][sample["names"].index(name)] for name in arm.grip.names]
                         for sample,arm in zip(request[q]["actual"], target)])
            at=dict(tick=0,actual_arm=ar,actual_fingers=fg,target_fingers=fg,
                    contact_state=dict(block_pose_task=episode_start,phase="approach",holders=[]))
        snap=dict(job=spec["job_id"],query=q,config=cfg,initial=start,at=at,
                  request=dict(request[q],stamp=dict(request[q]["stamp"],tick=at["tick"],sim_s=at["tick"]*.004)),
                  chunk=np.load(data/f"action-{q:05}.npy",allow_pickle=False),
                  action_sha256=report["query_stats"][-1]["action_sha256"], row=row,
                  failure=result["failure"], last_plan=last_plan)
        if last_plan:
            f=data/last_plan["trajectory_file"]
            with np.load(f,allow_pickle=False) as plan:
                snap["trajectory"]={k:plan[k] for k in plan.files}
        (snapshots/f"{label}.json").write_text(json.dumps(snap,default=lambda x:x.tolist(),allow_nan=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("archive","assets","audit","output"):
        parser.add_argument(field,type=Path)
    parser.add_argument("--snapshots",type=Path)
    args=parser.parse_args()
    if args.snapshots:
        args.snapshots.mkdir(exist_ok=True)
    results=read(STUDY/"submission_receipts.json")["candidates"]
    report={label:audit_trial(label,spec,args.archive,args.assets,args.audit,args.snapshots)
            for label,spec in results.items()}
    with args.output.open("w") as file:
        json.dump(report,file,default=lambda x:x.tolist(),allow_nan=False,separators=(",",":"))
        file.write("\n")


if __name__=="__main__":
    main()
