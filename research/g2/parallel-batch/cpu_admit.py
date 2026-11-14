"""No CUDA or model. Exercise native scenes and the new protected scene binding."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from g2_adapter import Stamp
from g2_author_h50_limits import BUDGET, TrialClock
from g2_author_native_entry import author_scene, choose_horizon
from g2_author_native_h50 import (
    ARCHIVE_CONFIG,
    locked_json,
    native_guard,
    run_diagnostic,
)
from g2_author_native_h50_cpu import Headless
from g2_assets import load_arms
from g2_capture import capture_arms
from g2_initial import capture_initial, digest_json
from g2_parallel_spec import cache_row, load
from sapien.physx import PhysxRigidDynamicComponent


class CPUWitnessEnd(Exception):
    pass


class CPUModel:
    """Two verified requests with one safe synthetic response."""

    def __init__(self, valley):
        self.requests = []
        self.valley = valley

    def infer(self, request):
        from pace_runtime import validate_request

        validate_request(request)
        self.requests.append(
            dict(
                prompt=request["prompt"],
                state=request["state"].tolist(),
                cameras={
                    k: dict(shape=list(v.shape), dtype=str(v.dtype))
                    for k, v in request["images"].items()
                },
            )
        )
        if len(self.requests) == 2:
            raise CPUWitnessEnd()
        actions = np.tile(request["state"], (50, 1))
        actions[:, 0] = 0.009 * np.sin(np.arange(50) * 0.35)
        actions[:, 7] = 0.007 * np.sin(np.arange(50) * 0.35)
        if self.valley:
            actions[:, 0] = 0.015 * (1 - np.cos(np.arange(50) * 2 * np.pi / 36))
            actions[:, 7] = request["state"][7]
        return {"actions": actions}


def check_reset(pace, release, env, spec):
    from pace.robotwin import load_cached_scenes

    row, _ = cache_row(spec, load_cached_scenes)
    cfg = locked_json(release / "configs/unified-seed0-native-w50.json", ARCHIVE_CONFIG)
    arms = load_arms(
        pace / "assets",
        release / "asset-audit.json",
        "aloha-agilex",
        cfg["calibration"]["aloha-agilex"],
    )
    stamp = Stamp(spec["episode_id"], 0, 0.0)
    samples = capture_arms(env.robot, stamp)
    arms = tuple(replace(a, instance=s.instance) for a, s in zip(arms, samples))
    initial = capture_initial(
        env,
        arms,
        np.eye(4),
        spec["episode_id"],
        digest_json(row),
        "author-native-h50-reset",
        False,
    )
    guard = native_guard(env, arms, samples, cfg)
    from g2_adapter import read_snapshot, Snapshot

    q, fingers, _, tools = read_snapshot(Snapshot(stamp, samples, {}, np.eye(4)), arms)
    world = guard.collision
    world.refresh()
    if not world.measured_clear(q, fingers, check_unheld=True) or env.check_success():
        raise ValueError("invalid initial geometry or already successful reset")
    return dict(
        cache={"sha256": spec["scene_cache_sha256"]},
        scene_seed=spec["scene_seed"],
        episode_info=row["episode_info"],
        prompt=env.get_instruction(),
        block_mass_kg=float(
            env.box.actor.find_component_by_type(PhysxRigidDynamicComponent).mass
        ),
        initial=initial,
        contact_manifold_count=len(env.scene.get_contacts()),
        initial_geometry_pass=True,
        raw_joint_state=q.tolist(),
        tcp_task=tools.tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pace", "release", "config", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--valley", action="store_true")
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only probe requires hidden CUDA")
    pace, release, output = (x.resolve() for x in (a.pace, a.release, a.output))
    spec = load(a.config)
    output.mkdir()
    report = dict(
        id=spec["id"],
        method=spec["method"],
        mode=spec["mode"],
        config_sha256=sha256(a.config.read_bytes()).hexdigest(),
        model="cpu_substitute",
    )
    model = CPUModel(a.valley)
    adapter = Headless(output)
    if spec["mode"] == "author_native":
        with author_scene(pace, release, configure_cpu=adapter, scene=spec) as (
            env,
            run,
        ):
            reference = check_reset(pace, release, env, spec)
            choose, basis = choose_horizon(pace, spec)
            selected = []
            try:
                run(
                    env,
                    model,
                    choose,
                    lambda step, _, horizon: selected.append([step, horizon]),
                )
            except CPUWitnessEnd:
                pass
            if (
                len(model.requests) != 2
                or len(selected) != 1
                or env.take_action_cnt != selected[0][1]
            ):
                raise ValueError("upstream second request is at the wrong boundary")
            report.update(
                reset=reference,
                selected=selected,
                selector=basis,
                second_query_after_actions=env.take_action_cnt,
                cleanup=True,
            )
    else:
        started = monotonic()
        budget = replace(BUDGET, physics_steps=3)
        receipt = run_diagnostic(
            pace,
            release,
            a.reference.resolve(),
            output / "protected",
            spec["episode_id"],
            lambda: nullcontext(model),
            clock=TrialClock(budget, started, started + budget.hard_s),
            configure_cpu=adapter,
            scene=spec,
            reference_sha256=sha256(a.reference.read_bytes()).hexdigest(),
        )
        if (
            receipt["category"] != "resource_censored"
            or receipt["physics_steps"] != 3
            or receipt["resource_reason"] != "physics_steps"
        ):
            raise RuntimeError(json.dumps(receipt, indent=2))
        report.update(receipt=receipt, protected_cpu_budget_injected=True)
    report["requests"] = model.requests
    report["admission"] = "PASS"
    (output / "cpu-admission.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
