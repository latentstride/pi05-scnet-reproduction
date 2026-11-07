#!/usr/bin/env python3
"""CPU-only attribution of preserved 127147943 logs. No model or simulator imports."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np

from g2_assets import load_arms
from g2_kinematics import pose_error
from g2_native_expert_cpu import unpack_trace

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "research/g2/unified-transfer-review/g2-unified-transfer-candidate"


def digest(path):
    with path.open("rb") as stream:
        h = sha256()
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        for prefix in ("/public/home/", "/Users/"):
            if prefix in value:
                return "<PRIVATE_PATH>/" + value.split("/")[-1]
    return value


def relative_error(anchor, current):
    e = pose_error(np.asarray(anchor), np.asarray(current))
    return [
        float(np.linalg.norm(e[:3]) * 1000),
        float(np.rad2deg(np.linalg.norm(e[3:]))),
    ]


def forces(sample, period):
    """Scalar sum of contact impulse magnitudes / dt, not actuator force or Coulomb fit."""
    result = {}
    for contact in sample["contacts"]:
        if not any("box" in name.lower() for name in contact["body_names"]):
            continue
        key = "|".join(contact["body_names"])
        result[key] = (
            sum(float(np.linalg.norm(p["impulse"])) for p in contact["points"]) / period
        )
    return result


def mapping_audit(episode, config):
    """Reconstruct FK on saved input states and approved IK waypoint endpoints."""
    from g2_initial import matrix7

    assets = ROOT / "research/robotwin/embodiment_assets"
    audit = RELEASE / "asset-audit.json"
    source = load_arms(
        assets, audit, "aloha-agilex", config["calibration"]["aloha-agilex"]
    )
    target = load_arms(assets, audit, "ARX-X5", config["calibration"]["ARX-X5"])
    initial = json.loads((episode / "initial-readback.json").read_text())
    source_frames = [
        np.linalg.inv(matrix7(config["source_world_task"])) @ matrix7(b)
        for b in config["source_base_world"]
    ]
    target_frames = [
        np.linalg.inv(matrix7(initial["task_frame_world"]))
        @ matrix7(initial["robot_base_world"][side])
        for side in ("left", "right")
    ]
    endpoints = []
    for path in sorted(episode.glob("plan-*.npz")):
        plan = np.load(path, allow_pickle=False)
        action_path = episode / path.name.replace("plan-", "action-").replace(
            ".npz", ".npy"
        )
        actions = np.load(action_path, allow_pickle=False)
        for index, tick in enumerate(plan["waypoint_ticks"]):
            for side in (0, 1):
                goal = (
                    source_frames[side]
                    @ source[side].chain.fk(actions[index, side * 7 : side * 7 + 6])
                    @ source[side].offset()
                )
                actual = (
                    target_frames[side]
                    @ target[side].chain.fk(plan["q"][tick, side])
                    @ target[side].offset()
                )
                endpoints.append(
                    dict(
                        file=path.name,
                        waypoint=index,
                        side=side,
                        source_pose=goal.tolist(),
                        target_pose=actual.tolist(),
                        residual=relative_error(goal, actual),
                    )
                )
    latest = {
        side: np.asarray(
            target_frames[i]
            @ target[i].chain.fk(initial["actual_arm_qpos_rad"][side])
            @ target[i].offset()
        )
        for i, side in enumerate(("left", "right"))
    }
    observations = []
    for lineno, line in enumerate((episode / "events.jsonl").open(), 1):
        event = json.loads(line)
        if event.get("kind") == "executed_step":
            latest = dict(
                zip(("left", "right"), map(np.asarray, event["actual_tools_task"]))
            )
        if event.get("kind") == "policy_request":
            for side, name in enumerate(("left", "right")):
                state = event["state"][side * 7 : side * 7 + 6]
                reconstructed = (
                    source_frames[side]
                    @ source[side].chain.fk(state)
                    @ source[side].offset()
                )
                observations.append(
                    dict(
                        line=lineno,
                        tick=event["stamp"]["tick"],
                        side=side,
                        measured_pose=latest[name].tolist(),
                        reconstructed_pose=reconstructed.tolist(),
                        residual=relative_error(latest[name], reconstructed),
                    )
                )
    return dict(
        approved_waypoints=endpoints,
        observation_fk=observations,
        max_waypoint_position_mm=max(x["residual"][0] for x in endpoints),
        max_waypoint_angle_deg=max(x["residual"][1] for x in endpoints),
        max_observation_position_mm=max(x["residual"][0] for x in observations),
        max_observation_angle_deg=max(x["residual"][1] for x in observations),
    )


def analyze(raw, output):
    output.mkdir(parents=True, exist_ok=False)
    campaign = json.loads((raw / "supervisor/campaign.json").read_text())
    assert len(campaign["rows"]) == 10 and campaign["stop_reason"] is None
    assert campaign["model_server_starts"] == 1
    hashes = {
        str(f.relative_to(raw)): dict(sha256=digest(f), bytes=f.stat().st_size)
        for f in sorted(raw.rglob("*"))
        if f.is_file()
    }
    rows = []
    public = zipfile.ZipFile(output / "key-evidence.zip", "x", zipfile.ZIP_DEFLATED)
    for directory in sorted((raw / "trials").iterdir()):
        episode = directory / "episode"
        receipt = json.loads((directory / "receipt.json").read_text())
        result = receipt.get("result", receipt)
        counts = Counter()
        steps, samples, states = {}, {}, {}
        significant, requests, rejected, proxy = [], [], [], []
        transitions = []
        for lineno, line in enumerate((episode / "events.jsonl").open(), 1):
            event = json.loads(line)
            kind = event.get("kind", "step")
            counts[kind] += 1
            tick = event.get("tick")
            if kind in ("executed_step", "step"):
                steps[tick] = event
            if kind == "native_contact_sample":
                samples[tick] = event
            if kind == "native_guard_state":
                states[tick] = event
            contact = event.get("contact_state", {})
            if "holders" in contact and (
                not transitions or transitions[-1]["holders"] != contact["holders"]
            ):
                transitions.append(dict(tick=tick, holders=contact["holders"]))
            if kind == "native_action_request":
                requests.append(event)
            if kind == "native_guard_rejection":
                rejected.append(dict(line=lineno, event=event))
            if kind in ("original_proxy_rejection", "physical_proxy_recovery"):
                proxy.append(dict(line=lineno, event=event))
            if kind not in (
                "executed_step",
                "step",
                "native_contact_sample",
                "native_guard_state",
                "gripper_target",
            ):
                significant.append(dict(line=lineno, event=event))
        blocks = []
        for f in sorted(episode.glob("action-*.npy")):
            chunk = np.load(f, allow_pickle=False)
            assert chunk.shape == (50, 14) and np.isfinite(chunk).all()
            entry = dict(file=f.name, sha256=digest(f))
            for side, col in [("left", 6), ("right", 13)]:
                indices = np.flatnonzero(chunk[:, col] < 0.5)
                entry[side + "_first_close"] = int(indices[0]) if len(indices) else None
                entry[side + "_minimum"] = float(chunk[:, col].min())
                entry[side + "_prefix_close_count"] = int(
                    np.sum(
                        chunk[
                            : (
                                50
                                if "w50" in directory.name
                                else 31
                                if "w31" in directory.name
                                else 4
                            ),
                            col,
                        ]
                        < 0.5
                    )
                )
            blocks.append(entry)
            public.write(f, directory.name + "/" + f.name)
        row = dict(
            episode_id=directory.name,
            category=receipt["category"],
            success=result["success"],
            outcome=result["outcome"],
            queries=result.get("queries"),
            actions=result.get("executed", result.get("policy_actions")),
            failure=clean(result.get("failure")),
            elapsed_s=receipt["elapsed_s"],
            counts=dict(counts),
            transitions=transitions,
            blocks=blocks,
            close_request_scope="native_action_request events only; ARX prefix intents are in blocks",
            close_requests={
                side: [
                    r["action_index_in_episode"]
                    for r in requests
                    if r["normalized_qpos_request"][col] < 0.5
                ]
                for side, col in [("left", 6), ("right", 13)]
            },
            cleanup=receipt["cleanup"],
            process_group_reaped=campaign["rows"][directory.name][
                "process_group_reaped"
            ],
            rejections=clean(rejected),
            proxy_events=clean(proxy),
        )
        row["holder_transition_samples"] = [
            dict(
                transition=t,
                step=steps.get(t["tick"]),
                native_sample=samples.get(t["tick"]),
            )
            for t in transitions
        ]
        if steps:
            row["tracking_peak_mm"] = (
                max(max(s.get("position_error_m", [0])) for s in steps.values()) * 1000
            )
        slip = []
        held = [
            t for t, s in states.items() if s["contact_state"]["holders"] == ["left"]
        ]
        if held:
            first, last = held[0], max(steps)
            anchor = states[first]["contact_state"]["tool_block"]["left"]
            first_tool = np.asarray(steps[first]["actual_tools_task"][0])
            first_block = np.asarray(samples[first]["block_pose_task"])
            for tick in range(first, last + 1):
                step, sample = steps[tick], samples[tick]
                tool, block = (
                    np.asarray(step["actual_tools_task"][0]),
                    np.asarray(sample["block_pose_task"]),
                )
                displacement = relative_error(anchor, np.linalg.inv(tool) @ block)
                slip.append(
                    dict(
                        tick=tick,
                        sim_s=step["sim_s"],
                        slip_mm=displacement[0],
                        slip_deg=displacement[1],
                        actual_fingers_m=step["actual_fingers"][0],
                        drive_fingers_m=sample["target_fingers_drive_m"][0],
                        tool_change=relative_error(first_tool, tool),
                        block_change=relative_error(first_block, block),
                        block_z_m=float(block[2, 3]),
                        impulse_over_dt_n=forces(sample, step["period_s"]),
                    )
                )
            row["slip"] = dict(
                first=slip[0],
                last=slip[-1],
                dt_s=steps[first]["period_s"],
                holding_duration_s=(last - first) * steps[first]["period_s"],
                max_drive_opening_m=float(
                    np.diff([s["drive_fingers_m"][0] for s in slip]).max()
                ),
                first_translation_crossing=next(
                    (s["tick"] for s in slip if s["slip_mm"] > 5), None
                ),
                first_rotation_crossing=next(
                    (s["tick"] for s in slip if s["slip_deg"] > 5), None
                ),
                frozen_block_counterfactual=relative_error(
                    anchor,
                    np.linalg.inv(np.asarray(steps[last]["actual_tools_task"][0]))
                    @ first_block,
                ),
                frozen_tool_counterfactual=relative_error(
                    anchor,
                    np.linalg.inv(first_tool)
                    @ np.asarray(samples[last]["block_pose_task"]),
                ),
            )
        # Spatial envelopes by measured phase, without matching unrelated frame indices.
        phase_points = {}
        for tick, step in steps.items():
            contact = step.get(
                "contact_state", states.get(tick, {}).get("contact_state", {})
            )
            phase = "+".join(contact.get("holders", [])) or "no_holder"
            phase_points.setdefault(phase, []).append(
                np.asarray(step["actual_tools_task"])[:, :3, 3]
            )
        row["phase_tcp_ranges_m"] = {
            phase: dict(min=np.min(v, axis=0).tolist(), max=np.max(v, axis=0).tolist())
            for phase, v in phase_points.items()
        }
        for filename in ("collision-checks.jsonl", "proxy-collision-checks.jsonl"):
            path = episode / filename
            if not path.exists():
                continue
            blocked = []
            for lineno, line in enumerate(path.open(), 1):
                check = json.loads(line)
                if check.get("blocked"):
                    blocked.append(dict(line=lineno, check=check))
            row[filename] = dict(
                blocked_checks=len(blocked),
                first=clean(blocked[:1]),
                last=clean(blocked[-1:]),
            )
        evidence = dict(
            significant=significant,
            terminal_steps=[steps[t] for t in sorted(steps)[-3:]],
            slip=slip,
            receipt=receipt,
            held_raw=[
                dict(step=steps[t], sample=samples[t], state=states.get(t))
                for t in range(held[0], max(steps) + 1)
            ]
            if held
            else [],
        )
        public.writestr(
            directory.name + "/evidence.json",
            json.dumps(clean(evidence), ensure_ascii=False, separators=(",", ":"))
            + "\n",
        )
        if "arx-full" in directory.name:
            config = json.loads(
                (RELEASE / "configs" / (directory.name + ".json")).read_text()
            )
            row["mapping_audit"] = mapping_audit(episode, config)
        rows.append(row)
    public.close()
    # Reconstruct native expert FK at recorded source segment boundaries.
    expert = []
    for seed in (0, 1):
        cfg = json.loads(
            (RELEASE / f"configs/unified-seed{seed}-native-w50.json").read_text()
        )
        arms = load_arms(
            ROOT / "research/robotwin/embodiment_assets",
            RELEASE / "asset-audit.json",
            "aloha-agilex",
            cfg["calibration"]["aloha-agilex"],
        )
        base = json.loads(
            (
                raw
                / f"trials/unified-seed{seed}-native-w50/episode/initial-readback.json"
            ).read_text()
        )
        source = RELEASE / f"expert/seed{seed}"
        meta = json.loads((source / "input.json").read_text())
        trace = unpack_trace(source / "runtime-trace.npz")
        from g2_initial import matrix7

        frames = [
            np.linalg.inv(matrix7(base["task_frame_world"]))
            @ matrix7(base["robot_base_world"][side])
            for side in ("left", "right")
        ]
        boundaries = []
        for segment in meta["segments"]:
            tick = segment["last_tick"]
            q = trace["target"][tick]
            tcp = [
                (frame @ arm.chain.fk(q[offset : offset + 6]) @ arm.offset()).tolist()
                for frame, arm, offset in zip(frames, arms, (0, 7))
            ]
            boundaries.append(
                dict(
                    source_tick=tick,
                    kind=segment["kind"],
                    target=q.tolist(),
                    tcp_task=tcp,
                )
            )
        expert.append(
            dict(
                seed=seed,
                metadata_sha256=digest(source / "input.json"),
                trace_sha256=digest(source / "runtime-trace.npz"),
                scope="source command FK, not measured native replay",
                spawn=meta["spawn_pose"],
                actual_policy_initial=base["objects_task"],
                boundaries=boundaries,
            )
        )
    # Pairwise identity applies only before feedback divergence.
    comparisons = []
    for seed in (0, 1):
        dirs = {
            k: raw / f"trials/unified-seed{seed}-{k}/episode"
            for k in ("native-w31", "native-w50", "native-w4", "arx-full-w4")
        }
        first = {
            k: np.load(d / "action-00001.npy", allow_pickle=False)
            for k, d in dirs.items()
        }
        observations = {
            k: np.load(d / "observation-00001.npz", allow_pickle=False)
            for k, d in dirs.items()
        }
        image_diff = {
            cam: dict(
                equal=bool(
                    np.array_equal(
                        observations["native-w4"][cam], observations["arx-full-w4"][cam]
                    )
                ),
                mean_absolute_pixel_difference=float(
                    np.abs(
                        observations["native-w4"][cam].astype(float)
                        - observations["arx-full-w4"][cam]
                    ).mean()
                ),
            )
            for cam in observations["native-w4"]
        }
        comparisons.append(
            dict(
                seed=seed,
                native_first_block_equal=bool(
                    np.array_equal(first["native-w31"], first["native-w50"])
                ),
                native_arx_first_block_max_abs=float(
                    np.max(np.abs(first["native-w4"] - first["arx-full-w4"]))
                ),
                native_arx_images=image_diff,
            )
        )
    report = dict(
        job_id="127147943",
        gpu_used=False,
        simulation_run=False,
        raw_files=len(hashes),
        raw_bytes=sum(x["bytes"] for x in hashes.values()),
        model_starts=campaign["model_server_starts"],
        stop_reason=campaign["stop_reason"],
        trials=rows,
        source_expert=expert,
        startup=clean(campaign["startup"]),
        first_comparisons=comparisons,
        original_archive_sha256="79b6c322dae6614d5a2b5a1301b893ff33c361a8640604702cbeddeae4ee9d86",
    )
    assert sum(r["success"] for r in rows if "expert" in r["episode_id"]) == 2
    assert sum(r["success"] for r in rows if "expert" not in r["episode_id"]) == 0
    assert all(r["process_group_reaped"] and all(r["cleanup"].values()) for r in rows)
    for name, value in [("metrics.json", report), ("raw-manifest.json", hashes)]:
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            dict(
                files=len(hashes),
                trials=len(rows),
                policy_successes=0,
                expert_successes=2,
                output=str(output),
            )
        )
    )


def verify_public(directory):
    """Independently recompute slip, impulses, actions and FK residuals from excerpts."""
    metrics = json.loads((directory / "metrics.json").read_text())
    checked_slip = checked_fk = checked_blocks = 0
    with zipfile.ZipFile(directory / "key-evidence.zip") as archive:
        for row in metrics["trials"]:
            evidence = json.loads(archive.read(row["episode_id"] + "/evidence.json"))
            assert evidence["receipt"]["category"] == row["category"]
            for block in row["blocks"]:
                data = archive.read(row["episode_id"] + "/" + block["file"])
                assert sha256(data).hexdigest() == block["sha256"]
                from io import BytesIO

                chunk = np.load(BytesIO(data), allow_pickle=False)
                for side, col in [("left", 6), ("right", 13)]:
                    indices = np.flatnonzero(chunk[:, col] < 0.5)
                    first = int(indices[0]) if len(indices) else None
                    assert first == block[side + "_first_close"]
                checked_blocks += 1
            held = evidence["held_raw"]
            if held:
                anchor = held[0]["state"]["contact_state"]["tool_block"]["left"]
                for raw, expected in zip(held, evidence["slip"], strict=True):
                    actual = (
                        np.linalg.inv(raw["step"]["actual_tools_task"][0])
                        @ raw["sample"]["block_pose_task"]
                    )
                    np.testing.assert_allclose(
                        relative_error(anchor, actual),
                        [expected["slip_mm"], expected["slip_deg"]],
                        rtol=0,
                        atol=1e-9,
                    )
                    assert (
                        forces(raw["sample"], raw["step"]["period_s"])
                        == expected["impulse_over_dt_n"]
                    )
                    checked_slip += 1
                assert evidence["slip"][-1]["slip_mm"] > 5
                assert evidence["slip"][-1]["slip_deg"] < 5
            mapping = row.get("mapping_audit")
            if mapping:
                for point in mapping["approved_waypoints"]:
                    np.testing.assert_allclose(
                        relative_error(point["source_pose"], point["target_pose"]),
                        point["residual"],
                        rtol=0,
                        atol=1e-9,
                    )
                    checked_fk += 1
                for point in mapping["observation_fk"]:
                    np.testing.assert_allclose(
                        relative_error(
                            point["measured_pose"], point["reconstructed_pose"]
                        ),
                        point["residual"],
                        rtol=0,
                        atol=1e-9,
                    )
                    checked_fk += 1
    result = dict(
        passed=True,
        trials=len(metrics["trials"]),
        raw_action_blocks=checked_blocks,
        slip_ticks=checked_slip,
        fk_observation_and_endpoint_checks=checked_fk,
        gpu_used=False,
        simulation_run=False,
    )
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--verify-public", type=Path)
    args = parser.parse_args()
    if args.verify_public:
        verify_public(args.verify_public)
    elif args.raw is not None and args.output is not None:
        analyze(args.raw, args.output)
    else:
        parser.error("supply raw output, or --verify-public directory")
