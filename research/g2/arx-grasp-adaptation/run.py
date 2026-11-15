"""A/B/C counterfactual CPU PhysX replay from a single archived G1 query."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np

from g2_adapter import G2Error, deadline, read_snapshot
from g2_arx_direct import DirectAdapter
from g2_arx_endpoint_cpu import build_scene, save
from g2_arx_grasp_adaptation import FirstLeftGrasp, load_jaws
from g2_arx_transfer_probe_cpu import restart_backend
from robotwin_control_geometry import Status, inverse_pose


def run(args):
    spec = json.loads(args.input.read_text())
    cfg = spec["config"]
    args.output.mkdir()
    scene, old, initial, drive = build_scene(spec, args.audit, args.assets, args.output)
    backend, _, physics = restart_backend(scene, old, initial, drive, cfg, spec["initial"], args.assets)
    actors = {actor.name: actor for actor in scene.get_all_actors()}
    backend.env.table, backend.env.target_box = actors["table"], actors["target_box"]
    # Stop only on explicit local-grasp metrics, with no injected task success.
    backend.env.check_success = lambda: False
    adapter = DirectAdapter(old.aloha, old.arx, old.source_frames, np.eye(4), old.seed, old.source_speed,
                           target_collision=old.target_collision, motion=old.motion.motion,
                           grip_rule=old.grip_rule, grip_max_clip=old.grip_max_clip)
    box_pose = np.asarray(spec["at"]["contact_state"]["block_pose_task"])
    asset = {r["robot"]: args.assets / r["urdf_member"] for r in json.loads(args.audit.read_text())["robots"]}
    source = load_jaws(asset["aloha-agilex"], adapter.aloha[0]) if args.mode != "A" else None
    target = load_jaws(asset["ARX-X5"], adapter.arx[0]) if args.mode != "A" else None
    assist = (FirstLeftGrasp(adapter,source,target,[.03,.03,.1],pose_source="oracle_object_pose")
              if args.mode != "A" else None)
    block = np.asarray(spec["chunk"]).reshape(50,2,7)
    start = monotonic()
    check = deadline(args.timeout)

    def remaining():
        check()
        return args.timeout - (monotonic() - start)

    stats = dict(mode=args.mode, source_job=spec["job"], query=spec["query"], source_action_sha256=spec["action_sha256"],
                 input_sha256=sha256(args.input.read_bytes()).hexdigest(), input_tick=backend.tick,
                 pose_source="oracle_object_pose" if assist else None, online_inference=False,
                 threshold_m=backend.collision.limits.penetration_m,
                 before_close_max_xy_mm=0., peak_held_lift_mm=0., maximum_simultaneous_left_fingers=0,
                 bilateral_steps=0, single_finger_steps=0, stable_hold_s=0., maximum_tracking_position_mm=0.,
                 segments=0, completed_original_rows=[], first_close_tick=None, first_holder_tick=None,
                 initial_pose=box_pose, config=cfg, physics_filter=physics)
    row = 0
    phase = "start"
    consecutive = 0

    def log(event):
        nonlocal consecutive
        stream.write(json.dumps(dict(replay_phase=phase, archived_row=row, **event),
                                default=lambda v:v.tolist(),allow_nan=False) + "\n")
        state = event.get("contact_state")
        if not state or not event.get("physics_step_count"):
            return
        fingers = len(state["contact_fingers"]["left"])
        xy = 1000*np.linalg.norm((np.asarray(state["block_pose_task"]) - box_pose)[:2,3])
        aperture = adapter.arx[0].grip.read(np.asarray(event["actual_fingers"][0]))
        if stats["first_close_tick"] is None:
            stats["before_close_max_xy_mm"] = max(stats["before_close_max_xy_mm"],float(xy))
            if aperture <= .95:
                stats["first_close_tick"] = event["tick"]
        stats["maximum_simultaneous_left_fingers"] = max(stats["maximum_simultaneous_left_fingers"],fingers)
        stats["bilateral_steps"] += fingers == 2
        stats["single_finger_steps"] += fingers == 1
        stats["maximum_tracking_position_mm"] = max(stats["maximum_tracking_position_mm"],
                                                    1000*max(event.get("position_error_m",[0])))
        if "left" in state["holders"] and fingers == 2:
            if stats["first_holder_tick"] is None:
                stats["first_holder_tick"] = event["tick"]
            consecutive += 1
            stats["stable_hold_s"] = max(stats["stable_hold_s"],consecutive*backend.dt)
            dz=1000*(state["block_pose_task"][2][3]-box_pose[2,3])
            stats["peak_held_lift_mm"] = max(stats["peak_held_lift_mm"],dz)
        else:
            consecutive = 0

    def execute(plan):
        stats["segments"] += 1
        backend.execute(plan,log,remaining,stop_on_contact=False,stop_on_success=False)
        transition = backend.take_replan()
        if transition:
            log(dict(kind="backend_replan",transition=transition))

    def snapshot_state():
        snap = backend.capture_state()
        _, grips, _, tools = read_snapshot(snap,adapter.arx)
        return snap, grips, tools, backend.collision.tracker.block_pose.copy()

    def move(tools, grip):
        snap, _, _, _ = snapshot_state()
        plan, residual = assist.plan(snap,tools,grip,remaining)
        log(dict(kind="generated_goal",targets=tools,normalized_grip=grip,ik_residual=residual,plan_ticks=len(plan.q)-1))
        execute(plan)

    def adapted_pose(raw):
        return assist.raw_targets(raw)

    with gzip.open(args.output/"events.jsonl.gz","wt") as stream:
        try:
            if args.mode in {"A","B"}:
                for row,raw in enumerate(block,1):
                    phase = "frozen_counterfactual"
                    if args.mode == "A":
                        row_start = backend.tick
                        while True:
                            snap, _, _, _ = snapshot_state()
                            result=adapter.policy_row(raw.ravel(),snap,remaining)
                            if not result.ok:
                                raise G2Error(result.status,result.detail,**result.diagnostic)
                            log(dict(kind="archived_plan",diagnostic=result.diagnostic,plan_ticks=len(result.payload.q)-1))
                            execute(result.payload)
                            if not result.diagnostic.get("partial_policy_row"):
                                break
                            if (backend.tick-row_start)*backend.dt >= result.diagnostic["row_duration_limit_s"]:
                                raise G2Error(Status.PLAN_FAILED,"original row motion duration exhausted")
                    else:
                        _, _, _, live_block = snapshot_state()
                        goals = adapted_pose(raw)
                        goals[0] = assist.center_target(goals[0],live_block)
                        move(goals,np.clip(raw[:,6],0,1))
                    stats["completed_original_rows"].append(row)
                    if row % 5 == 0:
                        print(args.mode,row,backend.tick,stats["bilateral_steps"],flush=True)
            else:
                closure = next(i for i,r in enumerate(block) if r[0,6] <= .5)
                raw = block[0]
                goals = adapted_pose(raw)
                grip = np.clip(raw[:,6],0,1)
                _, _, present, live_block = snapshot_state()
                phase = "retreat_without_lateral_motion"
                desired = assist.center_target(present[0],live_block,standoff=True)
                distance = np.dot(desired[:3,3]-present[0][:3,3],present[0][:3,0])
                goals[0] = present[0].copy()
                goals[0,:3,3] += goals[0,:3,0]*min(0.,distance)
                move(goals,grip)
                phase="align_at_standoff"
                _, _, _, live_block = snapshot_state()
                goals[0] = assist.center_target(adapted_pose(raw)[0],live_block,standoff=True)
                move(goals,grip)
                phase="approach_aligned"
                _, _, _, live_block = snapshot_state()
                policy_goal = adapted_pose(block[closure])[0].copy()
                goals[0] = assist.center_target(policy_goal,live_block)
                move(goals,grip)
                phase="verify_closure"
                _, measured, live, live_block = snapshot_state()
                slots = (measured[0]-adapter.arx[0].grip.closed)/target.travel_m
                stats["closure_geometry"] = assist.open_clearance(live[0],live_block,slots)
                if not stats["closure_geometry"]["ready"] or stats["before_close_max_xy_mm"] > 5:
                    raise G2Error(Status.PLAN_FAILED,"aligned left jaw closure prerequisites failed")
                phase="contact_aware_close"
                row = closure+1
                grip[0] = float(np.clip(block[closure,0,6],0,1))
                goals[0] = live[0]
                move(goals,grip)
                if "left" not in backend.collision.tracker.holders:
                    raise G2Error(Status.PLAN_FAILED,"limited-force closure produced no measured left holder")
                phase="confirm_hold"
                for _ in range(20):
                    if stats["stable_hold_s"] >= .3:
                        break
                    move(goals,grip)
                if stats["stable_hold_s"] < .3:
                    raise G2Error(Status.PLAN_FAILED,"left hold did not reach 0.3 seconds")
                phase="lift_from_archived_vertical_intent"
                _, _, live, _ = snapshot_state()
                lift = max(0., max(adapted_pose(r)[0,2,3] for r in block[closure:])-policy_goal[2,3])
                stats["archived_vertical_intent_mm"] = 1000*lift
                goals[0] = live[0]
                goals[0,2,3] += lift
                move(goals,grip)
        except G2Error as error:
            stats["failure"] = dict(phase=phase,row=row,code=error.status.value,
                                    detail=str(error),diagnostic=error.diagnostic)
        finally:
            stop = backend.tick
            backend.hold()
            stats.update(physical_steps=stop-stats["input_tick"],end_tick=stop,halt_tick_unchanged=backend.tick==stop,
                         holder_transitions=backend.collision.tracker.events,
                         terminal_contact=backend.collision.tracker.snapshot(),elapsed_s=monotonic()-start)
            stats["local_grasp_pass"] = bool(stats["before_close_max_xy_mm"] <= 5
                and stats["bilateral_steps"] and stats["stable_hold_s"] >= .3 and stats["peak_held_lift_mm"] >= 50
                and not stats.get("failure"))
            save(args.output/"receipt.json",stats)
    print(args.mode,stats["physical_steps"],stats["bilateral_steps"],stats["peak_held_lift_mm"],
          stats.get("failure",{}).get("detail"),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input","audit","assets","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--mode",choices=("A","B","C"),required=True)
    parser.add_argument("--timeout",type=int,default=390)
    run(parser.parse_args())
