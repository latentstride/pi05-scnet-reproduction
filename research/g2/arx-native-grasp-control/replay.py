"""Compare native ARX grasp waypoints, their Aloha-coordinate round trip, and archived policy."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np

from g2_adapter import G2Error, deadline, read_snapshot, solve_pair
from g2_arx_direct import DirectAdapter, kinematic_pair, execute_prefix
from g2_arx_endpoint_cpu import build_scene, save
from g2_arx_transfer_probe_cpu import restart_backend
from g2_kinematics import pose_error


def resolved_tools(arms, frames, q):
    return np.array([frame @ arm.chain.fk(value) @ arm.offset()
                     for frame, arm, value in zip(frames, arms, q)])


def errors(reference, actual):
    values = np.array([pose_error(a, b) for a, b in zip(reference, actual)])
    return dict(position_mm=(np.linalg.norm(values[:, :3], axis=1)*1000).tolist(),
                angle_deg=np.rad2deg(np.linalg.norm(values[:, 3:], axis=1)).tolist())


def run(args):
    raw = args.input.read_bytes()
    spec = json.loads(raw)
    cfg = spec["config"]
    args.output.mkdir()
    scene, old, initial, drive = build_scene(spec, args.audit, args.assets, args.output)
    backend, _, physical = restart_backend(scene, old, initial, drive, cfg, spec["initial"], args.assets)
    actors = {actor.name: actor for actor in scene.get_all_actors()}
    backend.env.table, backend.env.target_box = actors["table"], actors["target_box"]
    backend.env.check_success = lambda: False  # This replay stops at grasp/lift.
    backend.env.get_obs = lambda: {"observation": {name: {"rgb": np.zeros((8, 8, 3), np.uint8)}
        for name in ("head_camera", "left_camera", "right_camera")}}
    backend.video = SimpleNamespace(frame=lambda _: None)
    adapter = DirectAdapter(old.aloha, old.arx, old.source_frames, np.eye(4), old.seed, old.source_speed,
        target_collision=old.target_collision, motion=old.motion.motion,
        grip_rule=old.grip_rule, grip_max_clip=old.grip_max_clip)
    started = monotonic()
    check = deadline(360)
    def remaining():
        check()
        return 360 - (monotonic() - started)

    world = adapter.target_collision
    start = np.asarray(spec["at"]["contact_state"]["block_pose_task"])[:3, 3]
    result = dict(mode=args.mode, input_sha256=sha256(raw).hexdigest(), start_tick=backend.tick,
                  max_tracking_mm=[0., 0.], contact_steps=[0, 0], both_left_steps=0,
                  first_contact=None, first_holder=None, row_receipts=[], completed_rows=0,
                  max_block_rise_mm=0., max_block_planar_displacement_mm=0.,
                  safety_profile=world.profile_id, physics_filter=physical,
                  rgb="CPU fixture", policy_inference=False)
    number = 0
    with (args.output / "events.jsonl").open("x") as stream:
        def log(event):
            stream.write(json.dumps(dict(row=number, **event), default=lambda x:x.tolist()) + "\n")
            state = event.get("contact_state")
            if not state or not event.get("physics_step_count"):
                return
            counts = [len(state["contact_fingers"][side]) for side in ("left", "right")]
            center = np.asarray(state["block_pose_task"])[:3, 3]
            shift = (center-start)*1000
            result["max_block_rise_mm"] = max(result["max_block_rise_mm"], float(shift[2]))
            result["max_block_planar_displacement_mm"] = max(
                result["max_block_planar_displacement_mm"], float(np.linalg.norm(shift[:2])))
            for i, count in enumerate(counts):
                result["contact_steps"][i] += bool(count)
            result["both_left_steps"] += counts[0] == 2
            if "position_error_m" in event:
                result["max_tracking_mm"] = np.maximum(result["max_tracking_mm"],
                    1000*np.array(event["position_error_m"])).tolist()
            if counts[0] and result["first_contact"] is None:
                event = dict(event)
                fields = ("tick", "actual_fingers", "target_fingers", "actual_tools_task", "target_tools_task",
                          "physical_contacts", "position_error_m", "orientation_error_deg", "contact_state")
                result["first_contact"] = dict(row=number, fingers=counts, block_shift_mm=shift,
                                               **{k:event.get(k) for k in fields})
            if "left" in state["holders"] and result["first_holder"] is None:
                result["first_holder"] = dict(row=number, tick=event["tick"], block_shift_mm=shift)

        try:
            if args.mode == "policy":
                number = spec["row"]
                original = np.asarray(spec["chunk"], dtype=np.float64)[number-1]
                snapshot, _ = backend.capture()
                result["original_action_sha256"] = spec["action_sha256"]
                result["original_row"] = original.tolist()
                prefix = execute_prefix(backend, snapshot, [original], adapter.policy_row, remaining,
                                        log, lambda: None)
                result.update(replay_stop=prefix.reason, completed_rows=prefix.completed)
            else:
                _, _, frames, _ = read_snapshot(initial, adapter.arx)
                seed = old.seed.copy()
                for number, row in zip(spec["reference"]["indices"], spec["reference"]["native_rows"]):
                    reference = np.array(row).reshape(2, 7)
                    snapshot = backend.capture_state()
                    world.refresh()
                    task_tools = resolved_tools(adapter.arx, frames, reference[:, :6])
                    if args.mode == "roundtrip":
                        fingers = np.array([arm.grip.target(float(norm)) for arm, norm in zip(adapter.aloha, reference[:, 6])])
                        seed, _ = solve_pair(adapter.aloha, adapter.source_frames, task_tools, seed, fingers,
                                             kinematic_pair, deadline(remaining()))
                        source_tools = resolved_tools(adapter.aloha, adapter.source_frames, seed)
                        policy_form = np.column_stack((seed, reference[:, 6])).ravel()
                        planned = adapter.policy_row(policy_form, snapshot, remaining)
                    else:
                        source_tools = task_tools
                        planned = adapter.expert_row(reference.ravel(), snapshot, remaining)
                    waypoint = dict(native_tick=number, grip_normalized=reference[:, 6].tolist(),
                                    source_roundtrip_error=errors(task_tools, source_tools))
                    result["row_receipts"].append(waypoint)
                    if not planned.ok:
                        raise G2Error(planned.status, planned.detail, **planned.diagnostic)
                    plan = planned.payload
                    final_tools = resolved_tools(adapter.arx, frames, plan.q[-1])
                    waypoint.update(mapped_error=errors(task_tools, final_tools),
                                    target_fingers=plan.fingers[-1], partial=planned.diagnostic.get("partial_policy_row",False))
                    log(dict(kind="approved_plan", receipt=planned.diagnostic))
                    backend.execute(plan, log, remaining, stop_on_contact=False)
                    reason = backend.take_replan()
                    if reason:
                        waypoint["replan"] = reason
                        # A holder transition interrupts execution before the terminal point is guaranteed.
                        # Check the measured snapshot before planning the next archived expert sample.
                        backend.capture_state()
                    result["completed_rows"] += 1
                    if result["first_holder"] and result["max_block_rise_mm"] >= 20:
                        result["replay_stop"] = "bilateral_grasp_and_lift"
                        break
        except (G2Error, ValueError) as error:
            result["error"] = dict(row=number, detail=str(error),
                status=error.status.value if isinstance(error,G2Error) else "invalid",
                diagnostic=getattr(error,"diagnostic",{}))
        finally:
            tick = backend.tick
            backend.hold()
            result.update(physical_steps=tick-result["start_tick"], terminal=world.tracker.snapshot(),
                          hold_same_tick=tick==backend.tick, elapsed_s=monotonic()-started)
            save(args.output / "receipt.json",result)
    print(args.mode, args.input.name, result["physical_steps"], result["completed_rows"],
          result.get("replay_stop"), (result.get("error") or {}).get("detail"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("input", "audit", "assets", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--mode", choices=("native", "roundtrip", "policy"), required=True)
    run(parser.parse_args())
