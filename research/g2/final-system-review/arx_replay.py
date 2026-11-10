"""Replay ARX candidate admission from exact recorded snapshots, without moving PhysX."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np

from g2_adapter import Adapter, G2Error, Snapshot, Stamp, deadline, solve_pair
from g2_capture import capture_arms
from g2_final_geometry_sweep_cpu import snapshot
from g2_motion import QuinticMotion
from g2_remaining_coordinator import RecoveryCoordinator
from g2_remaining_router import JointRouter
from g2_static_failure_replay import alternatives, build, matrix


EPISODE = "unified-seed1-arx-full-w4"


def digest(path):
    with path.open("rb") as stream:
        h = sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def build_adapter(config, assets, audit, initial, at, request, output):
    q, g = np.asarray(at["actual_arm"]), np.asarray(at["actual_fingers"])
    audited = load(audit)
    scene, arms, source, _, _, world, proxy, _, _ = build(
        config,
        audit,
        assets,
        snapshot(config, initial, q, g, audited),
        initial,
        np.asarray(at["contact_state"]["block_pose_task"]),
        output,
    )
    if at["contact_state"]["phase"] != "approach" or at["contact_state"]["holders"]:
        raise ValueError("static free-space reconstruction does not support holding")
    drive = np.asarray(at["target_fingers"])
    world.set_start(g, drive)
    entities = {e.name: e for e in scene.get_all_articulations()}
    stamp = Stamp(**request["stamp"])
    samples = capture_arms(
        SimpleNamespace(
            left_entity=entities["arx_left"], right_entity=entities["arx_right"]
        ),
        stamp,
    )
    arms = tuple(replace(a, instance=s.instance) for a, s in zip(arms, samples))
    snap = Snapshot(stamp, samples, {}, matrix(config["world_task"]))
    limits = config["control_limits"]
    motion = QuinticMotion(
        config["control_period_s"],
        limits["arm_velocity_rad_s"],
        limits["arm_acceleration_rad_s2"],
        limits["finger_velocity_m_s"],
        limits["finger_acceleration_m_s2"],
        config["collision_joint_step_rad"],
        config["collision_finger_step_m"],
        limits["source"],
        "archived_ARX_geometry",
        max_ticks=config["max_plan_ticks"],
    )
    adapter = Adapter(
        source,
        arms,
        [matrix(x) for x in config["source_base_world"]],
        matrix(config["source_world_task"]),
        np.asarray(request["state"]).reshape(2, 7)[:, :6],
        np.asarray(config["source_proxy_velocity_rad_s"]),
        proxy,
        world,
        JointRouter(motion, np.array([a.chain.limits for a in arms])),
        grip_rule=config["grip_rule"],
        grip_max_clip=config["grip_max_clip"],
    )
    fractions = [arm.grip.measure(f)["normalized"] for arm, f in zip(arms, g)]
    proxy.set_start(np.array([arm.grip.target(v) for arm, v in zip(source, fractions)]))
    adapter.last_stamp = stamp
    return scene, adapter, snap, drive


def summary_error(error):
    d = error.diagnostic
    return dict(
        status=error.status.value,
        detail=str(error),
        reason=d.get("reason"),
        geometry=d.get("geometry"),
        collision_world=d.get("collision_world"),
        rejected_pairs=d.get("rejected_pairs"),
        continuity_rejected=d.get("continuity_rejected"),
        requested_rows=len(d.get("requests", ())),
        pair_depths_m=[
            c["max_penetration_m"]
            for ex in d.get("collision_examples", ())
            for c in ex["collisions"]
            if "max_penetration_m" in c
        ],
    )


def replay(run, release, assets, publication, output):
    folder = run / "trials" / EPISODE / "episode"
    published = load(publication)
    source_hash = published["source_sha256"]
    for leaf in ("events.jsonl", "action-00035.npy", "action-00036.npy", "result.json"):
        path = folder / leaf
        if digest(path) != source_hash[f"trials/{EPISODE}/episode/{leaf}"]:
            raise ValueError(f"archived source differs: {leaf}")
    proxy_log = folder / "proxy-collision-checks.jsonl"
    if digest(proxy_log) != source_hash[f"trials/{EPISODE}/episode/{proxy_log.name}"]:
        raise ValueError("archived proxy source differs")
    with proxy_log.open() as stream:
        logged_proxy = [json.loads(line) for line in stream]
    original = load(folder / "result.json")
    config_path = release / "configs" / f"{EPISODE}.json"
    config = load(config_path)
    audit = release / "asset-audit.json"
    initial = original["initial"]
    requests, at, step = [], {}, None
    with (folder / "events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "executed_step":
                step = row
            elif row["kind"] == "policy_request":
                requests.append(row)
                if len(requests) in (35, 36):
                    if step["tick"] != row["stamp"]["tick"]:
                        raise ValueError("missing same-tick readback")
                    at[len(requests)] = step
    reports = []
    for query in (35, 36):
        out = output / str(query)
        out.mkdir()
        scene, adapter, snap, drive = build_adapter(
            config,
            assets,
            audit,
            initial,
            at[query],
            requests[query - 1],
            out,
        )
        chunk = np.load(folder / f"action-{query:05d}.npy", allow_pickle=False)
        runner = RecoveryCoordinator(
            adapter,
            SimpleNamespace(_drive_fingers=lambda: drive),
            max_recoveries=config.get("max_recoveries", 4),
            approach_repair=True,
        )
        events = []
        report = dict(
            query=query, tick=snap.stamp.tick, original_source_q=adapter.seed.tolist()
        )
        failed_source = []

        def audit_solve(arms, frames, tools, seed, fingers, world, check, max_delta):
            try:
                return solve_pair(
                    arms, frames, tools, seed, fingers, world, check, max_delta
                )
            except G2Error:
                failed_source.append(
                    (arms, frames, tools, seed, fingers, world, max_delta)
                )
                raise

        with patch("g2_bounded_approach.solve_pair", audit_solve):
            try:
                result, _ = runner.action_plan(
                    chunk, snap, 4, lambda: 90, events.append
                )
                report["production_w4"] = dict(
                    approved=len(result.waypoint_ticks),
                    recovery_budget_used=runner.recoveries,
                    reason=events[-1]["kind"] if events else None,
                )
            except G2Error as error:
                report["production_w4"] = dict(approved=0, **summary_error(error))
        search = []
        for arms, frames, tools, seed, fingers, world, max_delta in failed_source:
            world.set_start(fingers)
            try:
                answer = alternatives(
                    arms, frames, tools, seed, fingers, world, max_delta
                )
            except G2Error as error:
                answer = summary_error(error)
            search.append(
                dict(
                    requested_tools=tools.tolist(),
                    max_delta_rad=max_delta.tolist(),
                    result=answer,
                )
            )
        report["bounded_extra_seed_search"] = search
        # Same-state geometry only, including the candidates originally filtered by continuity.
        evidence = (
            published["source_prefix_barriers"][0]["source_diagnostic"]
            if query == 35
            else original["failure"]["diagnostic"]
        )
        proxy = adapter.source_collision
        finger_state = proxy.initial_fingers
        counter = Counter()
        recorded_pair = []
        for verdict in evidence["candidate_verdicts"]:
            saved = next(
                (row for row in logged_proxy if row["q"] == verdict["candidate_q"]),
                None,
            )
            fingers = np.asarray(saved["fingers"]) if saved else finger_state
            accepted = proxy.ik_clear(
                np.asarray(verdict["candidate_q"]), fingers, deadline(10)
            )
            label = "clear" if accepted else "collision"
            counter[f"{verdict['reason']}_then_static_{label}"] += 1
            if verdict["reason"] == "collision":
                recorded_pair.extend(c for c in proxy.last_collisions)
        report["recorded_pair_geometry"] = dict(
            counts=counter, rechecked_collisions=recorded_pair
        )
        report["offline_window"] = {
            str(window): dict(
                raw_rows=window,
                first_gripper_at_or_below_0_5_one_based=[
                    next(
                        (i + 1 for i, v in enumerate(chunk[:window, idx]) if v <= 0.5),
                        None,
                    )
                    for idx in (6, 13)
                ],
                same_prefix_barrier_before_longer_rows=True,
                actual_adapter_window_valid=window == 4,
            )
            for window in (4, 26, 31, 50)
        }
        reports.append(report)
        print("ARX_CPU", query, report["production_w4"], flush=True)
        del scene
    audit_aloha = next(x for x in load(audit)["robots"] if x["robot"] == "aloha-agilex")
    urdf = assets / audit_aloha["urdf_member"]
    root = ElementTree.parse(urdf).getroot()
    links = {}
    for name in ("fl_link4", "fl_link6"):
        node = root.find(f"link[@name='{name}']/collision")
        mesh = (urdf.parent / node.find("geometry/mesh").get("filename")).resolve()
        parent = next(
            j for j in root.findall("joint") if j.find("child").get("link") == name
        )
        links[name] = dict(
            parent=parent.find("parent").get("link"),
            joint=parent.get("name"),
            origin=node.find("origin").attrib,
            mesh=mesh.name,
            mesh_sha256=digest(mesh),
        )
    return dict(
        physics_steps=0,
        model_inference=False,
        urdf_sha256=digest(urdf),
        config_sha256=digest(config_path),
        links=links,
        reports=reports,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "release", "assets", "publication", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    result = replay(args.run, args.release, args.assets, args.publication, args.output)
    (args.output / "arx-replay.json").write_text(json.dumps(result, indent=2) + "\n")
