"""Headless execution of already saved policy blocks, never a new model query."""

from types import MethodType, SimpleNamespace as NS
import json
import time
import numpy as np
from g2_adapter import G2Error
from g2_episode import json_value, native_prefix
from g2_native_backend import NativeBackend


def run_context(ctx):
    args = ctx["args"]
    env = ctx["env"]
    scene = ctx["scene"]
    guard = ctx["guard"]
    arms = ctx["arms"]
    ctx["reset"]()
    env.take_action_cnt = 0
    env.scene = scene
    env.box = ctx["block"]
    env.get_obs = lambda: None
    env.take_action = MethodType(ctx["take_action"], env)
    env.close_env = lambda **kw: None
    video = NS(
        frame=lambda obs: None, close=lambda: None, path=args.output / "no-render-video"
    )
    enabled = getattr(args, "saved_approach_repair", False)
    backend = NativeBackend(
        env,
        arms,
        np.eye(4),
        ctx["initial"],
        video,
        guard=guard,
        preload_hold=getattr(args, "production_contact_preload", False),
        contact_preload=getattr(args, "production_contact_preload", False),
        approach_repair=enabled,
    )
    backend.episode = ctx["name"]
    backend.window = args.window
    records = []
    completed = 0
    failure = None
    barrier = None
    prefix_q_error = prefix_finger_error = prefix_block_error = 0.0
    first_change = None
    steps, contacts = ctx["steps"], ctx["contacts"]

    def log(event):
        nonlocal first_change, prefix_q_error, prefix_finger_error, prefix_block_error
        event = dict(event)
        if event.get("physics_step_count") and event.get("kind") is None:
            event["kind"] = "executed_step"
        if event.get("kind") == "native_approach_projection" and first_change is None:
            first_change = event["tick"]
        tick = event.get("tick")
        if tick in steps and (first_change is None or tick <= first_change):
            if event.get("kind") == "executed_step":
                prefix_q_error = max(
                    prefix_q_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(event["actual_arm"])
                                - steps[tick]["actual_arm"]
                            )
                        )
                    ),
                )
                prefix_finger_error = max(
                    prefix_finger_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(event["actual_fingers"])
                                - steps[tick]["actual_fingers"]
                            )
                        )
                    ),
                )
            if event.get("kind") == "native_contact_sample":
                prefix_block_error = max(
                    prefix_block_error,
                    float(
                        np.linalg.norm(
                            np.array(event["block_pose_task"])[:3, 3]
                            - np.array(contacts[tick]["block_pose_task"])[:3, 3]
                        )
                    ),
                )
        records.append(event)

    check = ctx["check"]
    started = time.monotonic()
    # A new request after control divergence is not synthesized. Only existing
    # blocks with explicitly recorded GPU request times are eligible.
    count = getattr(args, "saved_actions", 27)
    actions = []
    for block in sorted(ctx["raw"].glob("action-*.npy")):
        actions.extend(np.load(block, allow_pickle=False))
        if len(actions) >= count:
            break
    if len(actions) < count:
        raise ValueError("insufficient saved model actions")
    try:
        for i in range(0, count, args.window):
            chunk = np.array(actions[i : i + 50])
            # This helper accepts only full original saved blocks. No padded
            # predictions are manufactured, even for a finite prefix probe.
            if len(chunk) != 50:
                raise ValueError("saved replay requires an intact 50-row block")
            prefix = native_prefix(
                chunk,
                min(args.window, count - i),
                backend.native_limits,
                ctx["config"]["native_grip_max_clip"],
                log,
            )
            completed += backend.execute(prefix, log, check)
            reason = backend.take_replan()
            if reason is not None:
                from dataclasses import asdict

                barrier = asdict(reason)
                # Never execute a historical next block as if it came from the
                # changed physical state. Stop for a future real model request.
                break
    except G2Error as exc:
        completed += exc.diagnostic.get("completed_actions", 0)
        failure = dict(
            status=exc.status.value, detail=str(exc), diagnostic=exc.diagnostic
        )
    finally:
        backend.hold()
        backend.close(args.output)
    final_q = ctx["entity"].get_qpos()
    value = dict(
        cpu_only=True,
        model_inference=False,
        renderer=False,
        scope="saved policy request replay; no synthesized subsequent inference",
        saved_actions_requested=count,
        completed_actions=completed,
        physical_steps=backend.tick,
        approach_repair=enabled,
        production_contact_preload=getattr(args, "production_contact_preload", False),
        first_approach_correction_tick=first_change,
        failure=failure,
        requires_fresh_model_request=barrier is not None,
        approach_barrier=barrier,
        elapsed_s=time.monotonic() - started,
        unchanged_prefix_errors=dict(
            joint_rad=prefix_q_error,
            fingers_m=prefix_finger_error,
            block_m=prefix_block_error,
        ),
        final_contact_state=guard.collision.tracker.snapshot(),
        final_native_q=final_q,
    )
    (args.output / "saved-action-events.jsonl").write_text(
        "".join(
            json.dumps(r, default=json_value, allow_nan=False) + "\n" for r in records
        )
    )
    (args.output / "saved-action-result.json").write_text(
        json.dumps(value, default=json_value, allow_nan=False, indent=2) + "\n"
    )
    print("CPU_SAVED_ACTION_RESULT", json.dumps(value, default=json_value), flush=True)
    return value
