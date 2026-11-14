"""Compare the historical proxy rejection with the direct ARX CPU path."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arx_replay import build_adapter, EPISODE, load
from g2_adapter import G2Error, deadline
from g2_arx_direct import DirectAdapter, execute_prefix
from g2_arx_transfer_probe_cpu import restart_backend
from g2_bounded_approach import checked_prefix
from g2_proxy_collision import ProxyCollisionWorld
from g2_episode import save_json
from itertools import combinations
from g2_initial import digest_json


def run(args):
    raw = args.run / "trials" / EPISODE / "episode"
    cfg = load(args.release / "configs" / f"{EPISODE}.json")
    initial = load(raw / "result.json")["initial"]
    n = 0
    with (raw / "events.jsonl").open() as events:
        for line in events:
            e = json.loads(line)
            if e["kind"] == "executed_step":
                at = e
            if e["kind"] == "policy_request":
                n += 1
                if n == args.query:
                    request = e
                    break
    args.output.mkdir()
    scene, old, snap, drive = build_adapter(
        cfg,
        args.assets,
        args.release / "asset-audit.json",
        initial,
        at,
        request,
        args.output,
    )
    proxy = old.source_collision
    model = proxy.base.models[0]
    links = sorted({obj.name for obj in model.get_fcl_model().get_collision_objects()})
    profile = dict(
        scope="same_articulation_only",
        policy="arx_aloha_proxy_self_collision_record_only",
        pairs=list(combinations(links, 2)),
    )
    profile["sha256"] = digest_json(profile)
    proxy.base.use_self_profile(profile)
    view = SimpleNamespace(actual_scene=scene, world=proxy.base.world)
    proxy.refresh = lambda: ProxyCollisionWorld.refresh(view)
    backend, _, _ = restart_backend(scene, old, snap, drive, cfg, initial, args.assets)
    actors = {a.name: a for a in scene.get_all_actors()}
    backend.env.table, backend.env.target_box = actors["table"], actors["target_box"]
    backend.env.get_obs = lambda: {
        "observation": {
            name: {"rgb": np.arange(192, dtype=np.uint8).reshape(8, 8, 3)}
            for name in ("head_camera", "left_camera", "right_camera")
        }
    }
    backend.env.check_success = lambda: False
    backend.video = SimpleNamespace(frame=lambda _: None)
    adapter = DirectAdapter(
        old.aloha,
        old.arx,
        old.source_frames,
        np.eye(4),
        old.seed,
        old.source_speed,
        target_collision=old.target_collision,
        motion=old.motion.motion,
        grip_rule=old.grip_rule,
        grip_max_clip=old.grip_max_clip,
    )
    chunk = np.load(raw / f"action-{n:05}.npy", allow_pickle=False)
    check = deadline(150)
    remaining = lambda: check() or 60.0
    receipt = dict(query=n, start_tick=backend.tick, seed_source="archived request")
    if n == 113:
        old_plan = old.actions(chunk, snap, 4, remaining())
        if old_plan.ok:
            _, barrier = checked_prefix(old, old_plan.payload, snap, check)
            receipt["legacy_barrier"] = barrier
        else:
            receipt["legacy_target_plan_error"] = old_plan.detail
    else:
        chunk = chunk[8:]
    snap, _ = backend.capture()
    observed = adapter.observe(snap, cfg["prompt"], remaining())
    receipt["direct_observation"] = dict(ok=observed.ok, detail=observed.detail)
    completed = []
    try:
        out = execute_prefix(
            backend,
            snap,
            chunk[: 6 if n == 113 else 1],
            adapter.policy_row,
            remaining,
            lambda e: None,
            lambda: completed.append(backend.tick),
        )
        receipt["reason"] = out.reason
    except G2Error as error:
        receipt["error"] = dict(
            status=error.status.value, detail=str(error), diagnostic=error.diagnostic
        )
    finally:
        tick = backend.tick
        backend.hold()
        receipt.update(
            end_tick=tick,
            hold_no_advance=backend.tick == tick,
            completed_row_ticks=completed,
        )
        save_json(args.output / "receipt.json", receipt)
    print(
        "COLLISION_CPU",
        n,
        completed,
        receipt.get("error", {}).get("detail"),
        "legacy barrier",
        bool(receipt.get("legacy_barrier")),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "release", "assets", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("query", type=int, choices=(14, 113))
    run(p.parse_args())
