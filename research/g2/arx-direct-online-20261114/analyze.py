"""Reproduce direct-controller GPU results from the original two SCNet runs."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_g2_arx_result import analyze, jsonl

JOBS = ("127999407", "127999421")
EPISODE = "unified-seed1-arx-full-w4"


def read(path):
    return json.loads(path.read_text())


def summarize(root, job):
    run = root / job
    trial = run / "trials" / EPISODE
    folder = trial / "episode"
    receipt = read(trial / "receipt.json")
    result = receipt["result"]
    base = analyze(run, EPISODE)
    archive = read(run / "archive-manifest.json")
    for relative, record in archive.items():
        local = run / relative
        if local.is_file():
            assert local.stat().st_size == record["bytes"], relative
            assert sha256(local.read_bytes()).hexdigest() == record["sha256"], relative
    counts = Counter()
    phases = Counter()
    contacts = Counter()
    holders = Counter()
    requests = []
    row_groups = []
    fresh_boundaries = []
    physical_steps = 0
    peak_position = np.zeros(2)
    peak_angle = np.zeros(2)
    finger_min = np.full((2, 2), np.inf)
    min_distance = np.full(2, np.inf)
    block_min = np.full(3, np.inf)
    block_max = np.full(3, -np.inf)
    last_tick = None
    terminal_row = None
    max_contact_fingers = {side: 0 for side in ("left", "right")}
    drive_min = np.full((2, 2), np.inf)
    measured_contact_markers = []
    initial = None
    for line, event in jsonl(folder / "events.jsonl"):
        assert event is not None
        kind = event.get("kind")
        counts[kind] += 1
        if kind == "policy_request":
            requests.append(event)
            row_groups.append([])
            if initial is None:
                initial = event
        if kind == "direct_row_planning":
            terminal_row = event["row_in_prefix"]
        if kind == "approved_plan":
            row_groups[-1].append(event["row_in_prefix"])
        if kind == "direct_prefix_complete":
            fresh_boundaries.append(event)
        steps = event.get("physics_step_count", 0)
        if not steps:
            continue
        physical_steps += steps
        last_tick = event["tick"]
        state = event.get("contact_state") or {}
        phases[state.get("phase")] += steps
        for side, finger_ids in state.get("contact_fingers", {}).items():
            max_contact_fingers[side] = max(max_contact_fingers[side], len(finger_ids))
            if finger_ids:
                contacts[side] += steps
                measured_contact_markers.append(
                    dict(
                        query=len(requests),
                        row=terminal_row,
                        tick=event["tick"],
                        side=side,
                        fingers=len(finger_ids),
                    )
                )
        for side in state.get("holders", []):
            holders[side] += steps
        peak_position = np.maximum(peak_position, event["position_error_m"])
        peak_angle = np.maximum(peak_angle, event["orientation_error_deg"])
        finger_min = np.minimum(finger_min, event["actual_fingers"])
        drive_min = np.minimum(drive_min, event["target_fingers"])
        pose = state.get("block_pose_task")
        if pose is not None:
            xyz = np.asarray(pose)[:3, 3]
            tools = np.asarray(event["actual_tools_task"])[:, :3, 3]
            min_distance = np.minimum(min_distance, np.linalg.norm(tools - xyz, axis=1))
            block_min = np.minimum(block_min, xyz)
            block_max = np.maximum(block_max, xyz)
    actions = []
    for number, request in enumerate(requests, 1):
        file = folder / f"action-{number:05}.npy"
        chunk = np.load(file, allow_pickle=False)
        assert chunk.shape == (50, 14)
        close = []
        for index in (6, 13):
            rows = np.flatnonzero(chunk[:, index] <= 0.5)
            close.append(int(rows[0] + 1) if rows.size else None)
        actions.append(
            dict(
                query=number,
                stamp=request["stamp"],
                first_row_le_0_5=close,
                approved_rows=row_groups[number - 1],
                lowest_approved_gripper_value=[
                    (
                        float(chunk[np.asarray(row_groups[number - 1]) - 1, i].min())
                        if row_groups[number - 1]
                        else None
                    )
                    for i in (6, 13)
                ],
                fresh_real_observation=dict(
                    arms=len(request.get("actual") or []) == 2,
                    objects=sorted((request.get("objects_task") or {}).keys()),
                    rgb_file=f"observation-{number:05}.npz",
                ),
                file_sha256=sha256(file.read_bytes()).hexdigest(),
                rgb_sha256=sha256(
                    (folder / f"observation-{number:05}.npz").read_bytes()
                ).hexdigest(),
            )
        )
    assert len(requests) == result["queries"]
    assert sum(map(len, row_groups)) == result["policy_actions"]
    assert receipt["candidate_stop"]["tick"] == last_tick == physical_steps
    phases_log = [
        row
        for _, row in jsonl(run / "model/phases.jsonl")
        if row and row["event"] == "end"
    ]
    startup = read(run / "startup" / EPISODE / "receipt.json")
    config = read(run / "release/configs" / f"{EPISODE}.json")
    assert config["arx_direct"] == "arx_direct_v1" and config["window"] == 50
    assert (
        "task_tcp_projection" not in config
        and "free_space_approach_repair" not in config
    )
    target = result["failure"]["diagnostic"]
    depths = [
        hit["depth_m"]
        for item in target["candidate_verdicts"]
        for hit in item.get("collisions", [])
        if "depth_m" in hit
    ]
    assert depths and target["collision_world"] == "target_collision"
    return dict(
        job_id=job,
        policy_seed=config["policy_rng_seed"],
        release_commit=read(run / "authorization-receipt.json")["commit"],
        authorization=read(run / "authorization-receipt.json"),
        startup=startup,
        termination=base["termination"],
        candidate_stop=receipt["candidate_stop"],
        counters=base["counters"],
        queries=actions,
        event_counts=counts,
        prefix_boundaries=fresh_boundaries,
        terminal_query=len(requests),
        first_rejected_row=terminal_row,
        physical_steps=physical_steps,
        contact_phase_ticks=phases,
        finger_contact_ticks=contacts,
        holder_ticks=holders,
        max_simultaneous_contact_fingers=max_contact_fingers,
        measured_contact_markers=measured_contact_markers,
        min_target_fingers_m=drive_min.tolist(),
        target_candidate_penetration_mm=[min(depths) * 1000, max(depths) * 1000],
        model_phase_s={
            r["stage"]: r["elapsed_s"] for r in phases_log if r["stage"] != "inference"
        },
        peak_tracking_position_m=peak_position.tolist(),
        peak_tracking_angle_deg=peak_angle.tolist(),
        min_actual_fingers_m=finger_min.tolist(),
        min_tcp_to_block_center_m=min_distance.tolist(),
        block_task_extent_m=dict(min=block_min.tolist(), max=block_max.tolist()),
        initial_request_state=initial["state"],
        collision_audits=base["collision_audits"],
        artifact_manifest=archive,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    slurm = (args.archive / "slurm.psv").read_text()
    results = {j: summarize(args.archive, j) for j in JOBS}
    output = dict(slurm=slurm, results=results)
    with args.output.open("x") as file:
        json.dump(output, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
