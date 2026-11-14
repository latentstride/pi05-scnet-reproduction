"""Recompute recorded policy/FK/IK/input facts, with no inference or simulation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from g2_assets import load_arms
from g2_kinematics import pose_error
from robotwin_control_geometry import pose_from_wxyz


EPISODE = "unified-seed1-arx-full-w4"


def load(p):
    return json.loads(p.read_text())


def pose(p):
    return pose_from_wxyz(p[:3], p[3:])


def tools(arms, frames, q):
    return np.array([f @ a.chain.fk(v) @ a.offset()
                     for a, f, v in zip(arms, frames, q)])


def residual(a, b):
    errors = np.array([pose_error(x, y) for x, y in zip(a, b)])
    return np.linalg.norm(errors[:, :3], axis=1) * 1000, np.rad2deg(
        np.linalg.norm(errors[:, 3:], axis=1))


def grip_schedule(chunk):
    return {str(t): [int(rows[0]+1) if len(rows) else None
                    for rows in (np.flatnonzero(chunk[:,d]<=t) for d in (6,13))]
            for t in (.99,.9,.5,.1)}


def main(a):
    a.output.mkdir(exist_ok=True)
    norm = load(a.source / "norm/robotwin/norm_stats.json")["norm_stats"]["state"]
    lo, hi = np.array(norm["q01"]), np.array(norm["q99"])
    expert = np.load(a.reference, allow_pickle=False)
    assert sha256(a.reference.read_bytes()).hexdigest() == "e3c734e7796cb963788d10a0f3476b16636328d5a3b2b3da8bfda095662971b2"
    result = dict(scope="immutable recorded actions; FK arithmetic only", jobs={}, aloha={})
    for group in ("G1", "G2", "G3"):
        chunks = sorted((a.source / group).glob("chunk-*.npy"))
        rows = []
        for file in chunks:
            x = np.load(file, allow_pickle=False)
            rows.append(dict(file=file.name, sha256=sha256(file.read_bytes()).hexdigest(),
                gripper_threshold_first_rows=grip_schedule(x),
                close_first_row=[int(i[0]+1) if len(i) else None
                    for i in (np.flatnonzero(x[:, d] <= .5) for d in (6, 13))]))
        actions = np.concatenate([np.load(p, allow_pickle=False) for p in chunks])
        observations=[]
        for file in sorted((a.source/group).glob('request-*.npz')):
            with np.load(file,allow_pickle=False) as request:
                state=request['state']
                observations.append(dict(file=file.name,sha256=sha256(file.read_bytes()).hexdigest(),
                    state=state.tolist(),prompt=str(request['prompt']),
                    rgb={k:dict(shape=list(request[k].shape),mean=float(request[k].mean()),
                                std=float(request[k].std())) for k in request.files if k.startswith('cam_')},
                    input_dimensions_outside_training_q01_q99_by_more_than_0_001=np.flatnonzero(
                        (state<lo-.001)|(state>hi+.001)).tolist()))
        result["aloha"][group] = dict(chunks=rows,observations=observations,
            state_like_action_quantile_outlier_dimensions=np.flatnonzero(
                ((actions < lo - .001) | (actions > hi + .001)).any(axis=0)).tolist(),
            note="action range is not an observation distribution; observed camera statistics do not establish visual feature equivalence")
    for job in ("127999407", "127999421"):
        run = a.archive / job
        archive_manifest=load(run/'archive-manifest.json')
        raw = run / "trials" / EPISODE / "episode"
        cfg = load(run / "release/configs" / f"{EPISODE}.json")
        source = load_arms(a.assets, a.audit, "aloha-agilex", cfg["calibration"]["aloha-agilex"])
        target = load_arms(a.assets, a.audit, "ARX-X5", cfg["calibration"]["ARX-X5"])
        frames = [np.linalg.inv(pose(cfg["source_world_task"])) @ pose(p)
                  for p in cfg["source_base_world"]]
        requests, last, approved = [], None, None
        obs_p, obs_r, ik_p, ik_r, interpolation_p = [], [], [], [], []
        waypoint_end = []
        measured_grip=[];drive_grip=[]
        with (raw / "events.jsonl").open() as stream:
            for line in stream:
                e = json.loads(line)
                if e["kind"] == "policy_request":
                    requests.append(e)
                    fr = [np.linalg.inv(np.asarray(e["world_task"])) @ np.asarray(v["world_base"])
                          for v in e["actual"]]
                    actual = tools(target, fr, np.array([v["qpos"][:6] for v in e["actual"]]))
                    predicted = tools(source, frames, np.asarray(e["state"]).reshape(2, 7)[:, :6])
                    p, r = residual(predicted, actual)
                    obs_p.append(p); obs_r.append(r)
                if e["kind"] == "approved_plan":
                    if approved is not None and last is not None:
                        p, r = residual(np.asarray(approved["receipt"]["requests"][0]["task_tools"]),
                                        np.asarray(last["actual_tools_task"]))
                        waypoint_end.append(dict(query=approved["query"], row=approved["row_in_prefix"],
                            tick=last["tick"], position_mm=p.tolist(), angle_deg=r.tolist()))
                    approved = dict(e, query=len(requests));last=None
                    request = e["receipt"]["requests"][0]
                    mapped = tools(target, fr, np.array(request["ik_target"]))
                    p, r = residual(np.array(request.get("execution_tools", request["task_tools"])), mapped)
                    ik_p.append(p);ik_r.append(r)
                if e.get("physics_step_count"):
                    interpolation_p.append(e["position_error_m"])
                    measured_grip.append([arm.grip.measure(np.array(v))['normalized']
                                          for arm,v in zip(target,e['actual_fingers'])])
                    drive_grip.append([arm.grip.read(np.array(v))
                                       for arm,v in zip(target,e['target_fingers'])])
                    last=e
        if approved is not None and last is not None:
            p, r = residual(np.asarray(approved["receipt"]["requests"][0]["task_tools"]),
                            np.asarray(last["actual_tools_task"]))
            waypoint_end.append(dict(query=approved["query"], row=approved["row_in_prefix"],
                tick=last["tick"], position_mm=p.tolist(), angle_deg=r.tolist()))
        # Expert reference belongs to the same scene, but it is native target
        # control, not policy transfer. Compare pose sets, never align row clocks.
        expert_closed = np.flatnonzero(expert["g1"][:, 0] <= .5)
        grasp_idx = int(expert_closed[0])
        expert_pregrasp = tools(target, fr, expert["q1"][grasp_idx])
        summaries=[]
        states=np.array([r["state"] for r in requests])
        for n, request in enumerate(requests, 1):
            file=raw/f"action-{n:05}.npy"
            assert sha256(file.read_bytes()).hexdigest()==archive_manifest[str(file.relative_to(run))]['sha256']
            block=np.array(request["objects_task"]["block"])
            chunk=np.load(file, allow_pickle=False)
            endpoints=np.array([tools(source, frames, row.reshape(2,7)[:,:6]) for row in chunk])
            distance=np.linalg.norm(endpoints[:,:,:3,3]-block[:3,3],axis=2)*1000
            closure=[]
            for side,d in enumerate((6,13)):
                indices=np.flatnonzero(chunk[:,d]<=.5)
                entry=dict(first_row=int(indices[0]+1) if len(indices) else None)
                if len(indices):
                    row=int(indices[0]); xyz,ang=residual(endpoints[row],expert_pregrasp)
                    entry.update(tcp_to_block_center_mm=float(distance[row,side]),
                        offset_block_local_mm=(block[:3,:3].T@(endpoints[row,side,:3,3]-block[:3,3])*1000).tolist(),
                        difference_from_expert_first_closure_tcp_mm=float(xyz[side]),
                        difference_from_expert_first_closure_rotation_deg=float(ang[side]))
                closure.append(entry)
            summaries.append(dict(query=n, tick=request["stamp"]["tick"],
                action_sha256=sha256(file.read_bytes()).hexdigest(),
                gripper_threshold_first_rows=grip_schedule(chunk),
                min_predicted_tcp_to_block_center_mm=distance.min(axis=0).tolist(),
                closest_row=(distance.argmin(axis=0)+1).tolist(), closure=closure,
                selected_rows=[dict(row=i+1,grippers=chunk[i,[6,13]].tolist(),
                    tcp_to_block_center_mm=distance[i].tolist(),
                    tcp_offset_block_local_mm=[(block[:3,:3].T@(p[:3,3]-block[:3,3])*1000).tolist()
                                                for p in endpoints[i]],
                    distance_from_expert_closure_tcp_mm=residual(endpoints[i],expert_pregrasp)[0].tolist())
                    for i in (0,7,25,26,34,35,49)],
                rgb={k:dict(shape=list(v.shape), mean=float(v.mean()),std=float(v.std()))
                     for k,v in np.load(raw/f"observation-{n:05}.npz").items()}))
        normalized=2*(states-lo)/(hi-lo+1e-6)-1
        result["jobs"][job]=dict(
            policy_seed=cfg["policy_rng_seed"], prompt=requests[0]["prompt"],
            query_summaries=summaries,
            max_observation_round_trip_position_mm=np.max(obs_p,axis=0).tolist(),
            max_observation_round_trip_angle_deg=np.max(obs_r,axis=0).tolist(),
            max_target_ik_position_mm=np.max(ik_p,axis=0).tolist(),
            max_target_ik_angle_deg=np.max(ik_r,axis=0).tolist(),
            max_physics_interpolated_tracking_position_mm=(np.max(interpolation_p,axis=0)*1000).tolist(),
            min_measured_aperture_fraction=np.min(measured_grip,axis=0).tolist(),
            min_executed_gripper_drive_fraction=np.min(drive_grip,axis=0).tolist(),
            max_raw_waypoint_endpoint_error_mm=np.max([w["position_mm"] for w in waypoint_end],axis=0).tolist(),
            waypoint_end=waypoint_end,
            quantile_input_range=[normalized.min(axis=0).tolist(),normalized.max(axis=0).tolist()],
            input_dimensions_outside_training_q01_q99_by_more_than_0_001=np.flatnonzero(
                ((states<lo-.001)|(states>hi+.001)).any(axis=0)).tolist(),
            input_state_min=states.min(axis=0).tolist(),input_state_max=states.max(axis=0).tolist(),
            expert_first_left_closure=dict(reference_index_zero_based=grasp_idx,
                native_grips=expert['g1'][grasp_idx].tolist(),tools=expert_pregrasp.tolist()),
        )
    (a.output/"chain-metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for job, r in result['jobs'].items():
        print(job, 'source round trip mm', r['max_observation_round_trip_position_mm'],
              'target IK mm',r['max_target_ik_position_mm'],
              'waypoint mm',r['max_raw_waypoint_endpoint_error_mm'])
        for q in r['query_summaries'][:2]:print(q['query'],q['min_predicted_tcp_to_block_center_mm'],q['closure'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','source','reference','assets','audit','output'):
        p.add_argument(key,type=Path)
    main(p.parse_args())
