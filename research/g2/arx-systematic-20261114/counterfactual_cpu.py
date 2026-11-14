"""Offline immutable policy replay: change observation boundary, keep ARX gates.

This is a diagnostic, never an online entry point. RGB is a labelled fixture;
there is no model call. A retained block is explicitly a counterfactual.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS
from time import monotonic

import numpy as np

from arx_replay import build_adapter, load
from g2_adapter import G2Error, deadline, read_snapshot
from g2_arx_direct import DirectAdapter, execute_prefix
from g2_arx_transfer_probe_cpu import restart_backend


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False,
                               default=lambda v: v.tolist()) + "\n")


def run(a):
    spec = load(a.input)
    a.output.mkdir()
    config = spec["config"]
    scene, old, snap, drive = build_adapter(
        config, a.assets, a.audit, spec["initial"], spec["at"],
        spec["request"], a.output,
    )
    backend, _, _ = restart_backend(scene, old, snap, drive, config,
                                     spec["initial"], a.assets)
    actors = {x.name: x for x in scene.get_all_actors()}
    backend.env.table, backend.env.target_box = actors["table"], actors["target_box"]
    backend.env.get_obs = lambda: {"observation": {
        k: {"rgb": np.arange(192, dtype=np.uint8).reshape(8, 8, 3)}
        for k in ("head_camera", "left_camera", "right_camera")}}
    backend.env.check_success = lambda: False
    backend.video = NS(frame=lambda _: None)
    adapter = DirectAdapter(old.aloha, old.arx, old.source_frames, np.eye(4),
        old.seed, old.source_speed, target_collision=old.target_collision,
        motion=old.motion.motion, grip_rule=old.grip_rule,
        grip_max_clip=old.grip_max_clip)
    chunk = np.asarray(spec["chunk"])
    start = backend.tick
    receipt = dict(job=spec["job"], query=spec["query"], mode=a.mode,
        action_sha256=spec["action_sha256"], start_tick=start,
        restart_scope="archived actual joints, fingers, drive and object pose; zero velocity; no solver cache",
        rgb="fixture; no model inference", stable_grasp=False,
        completed_rows=0, contacts={}, holders={}, max_fingers=[0, 0],
        safety="unchanged target ARX collision, contact, joints, motion and slip gates")
    started = monotonic()
    check = deadline(480)
    remaining = lambda: check() or 480 - (monotonic() - started)
    row_number = 0
    def consumed():
        receipt["completed_rows"] += 1
    with (a.output / "events.jsonl").open("x") as stream:
        def log(row):
            stream.write(json.dumps(dict(counterfactual_row=row_number, **row),
                default=lambda v: asdict(v) if hasattr(v, "__dataclass_fields__") else v.tolist()) + "\n")
            state = row.get("contact_state", {})
            for i, side in enumerate(("left", "right")):
                n = len(state.get("contact_fingers", {}).get(side, []))
                receipt["max_fingers"][i] = max(receipt["max_fingers"][i], n)
                if n:
                    receipt["contacts"][side] = receipt["contacts"].get(side, 0) + row.get("physics_step_count", 0)
            for side in state.get("holders", []):
                receipt["holders"][side] = receipt["holders"].get(side, 0) + row.get("physics_step_count", 0)
                receipt["stable_grasp"] = True
        try:
            snap, _ = backend.capture()
            if a.mode == "online_boundary":
                result = execute_prefix(backend, snap, chunk, adapter.policy_row,
                                        remaining, log, consumed)
                receipt["boundary"] = asdict(result)
            else:
                for row_number, row in enumerate(chunk, 1):
                    receipt["attempted_row"] = row_number
                    if row_number > 1:
                        snap = backend.capture_state()
                    q, fingers, _, _ = read_snapshot(snap, adapter.arx)
                    adapter.target_collision.refresh()
                    if not adapter.target_collision.measured_clear(q, fingers):
                        raise G2Error(__import__('robotwin_control_geometry').Status.COLLISION,
                            "counterfactual actual ARX state blocked",
                            collisions=adapter.target_collision.last_collisions)
                    planned = adapter.policy_row(row, snap, remaining)
                    if not planned.ok:
                        raise G2Error(planned.status, planned.detail, **planned.diagnostic)
                    log(dict(kind="counterfactual_approved", receipt=planned.diagnostic))
                    # No new observation boundary; every motion still gets its
                    # full production planning and per-physics-step safety checks.
                    done = backend.execute(planned.payload, log, remaining,
                                           stop_on_contact=False)
                    if done != 1:
                        raise RuntimeError("invalid checked row receipt")
                    consumed()
        except G2Error as error:
            receipt["error"] = dict(status=error.status.value, detail=str(error),
                                     diagnostic=error.diagnostic)
        finally:
            receipt["end_tick"] = backend.tick
            receipt["physical_steps"] = backend.tick - start
            backend.hold()
            receipt["hold_no_advance"] = backend.tick == receipt["end_tick"]
            save(a.output / "receipt.json", receipt)
    print("COUNTERFACTUAL", spec["job"], spec["query"], a.mode,
          receipt["completed_rows"], receipt.get("error", {}).get("detail"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "audit", "assets", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("mode", choices=("online_boundary", "retain_checked_block"))
    run(p.parse_args())
