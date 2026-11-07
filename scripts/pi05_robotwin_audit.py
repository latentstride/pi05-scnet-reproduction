#!/usr/bin/env python3
"""Read-only CPU audit of pinned public metadata and archived RoboTwin runs."""

import argparse
import csv
from hashlib import sha256
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G2 = "research/g2/"
AUDIT = G2 + "protocol-audit/"
HANDS = {"left": 6, "right": 13}
THRESHOLD = 0.5
SOURCES = "source-lock.json"
DATASET = "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0"


class Evidence:
    def __init__(self):
        self.hashes = {}

    def path(self, relative, root=ROOT):
        path = root / relative
        with path.open("rb") as stream:
            self.hashes[relative] = sha256(stream.read()).hexdigest()
        return path

    def read(self, relative, root=ROOT):
        return json.loads(self.path(relative, root).read_bytes())


def first_closed(chunk, column):
    rows = np.flatnonzero(chunk[:, column] < THRESHOLD)
    return int(rows[0]) if len(rows) else None


def close_counts(indices, lengths):
    present = sum(i is not None for i in indices)
    retained = sum(
        i is not None and i < k for i, k in zip(indices, lengths, strict=True)
    )
    return dict(predicted=present, retained=retained, truncated=present - retained)


def fetch_upstream(source_dir):
    lock = json.loads((ROOT / AUDIT / SOURCES).read_bytes())
    for item in lock["files"]:
        path = source_dir / item["file"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = subprocess.check_output(
                ["curl", "-fsSL", "--max-time", "35", item["url"]]
            )
            if sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError(f"unexpected upstream bytes: {item['file']}")
            path.write_bytes(raw)


def official_zip_recheck(source_dir, sample_dir):
    """Read the directory and three small ARX cache members via bounded HTTP ranges."""
    import h5py
    from g2_audit_expert_metadata import NumpyOnlyUnpickler, timeline
    from sample_robotwin_archives import RangeFile

    revision = json.loads((ROOT / AUDIT / SOURCES).read_bytes())["revisions"][
        "official_dataset"
    ]
    listing = json.loads((source_dir / "hf-data-handover_block.json").read_bytes())
    inventory, episodes = [], []
    for row in listing:
        if not row["path"].endswith(
            ("/demo_clean.zip", "/aloha-agilex_clean_50.zip", "/arx-x5_clean_50.zip")
        ):
            continue
        url = f"{DATASET}/resolve/{revision}/{row['path']}"
        remote = RangeFile(url, row["size"], 2 * 1024 * 1024)
        with zipfile.ZipFile(remote) as archive:
            infos = archive.infolist()
            inventory.append(
                dict(
                    archive=row["path"],
                    hdf5=[
                        dict(member=i.filename, bytes=i.file_size, crc32=f"{i.CRC:08x}")
                        for i in infos
                        if i.filename.endswith(".hdf5")
                    ],
                    planner_pkl_count=sum(
                        i.filename.endswith(".pkl") and "_traj_data/" in i.filename
                        for i in infos
                    ),
                    http_bytes=remote.transferred,
                    http_requests=remote.requests,
                )
            )
            if not row["path"].endswith("/arx-x5_clean_50.zip"):
                continue
            for ep in range(3):
                member = next(
                    i
                    for i in infos
                    if i.filename.endswith(f"/_traj_data/episode{ep}.pkl")
                )
                raw = archive.read(member)
                planner = NumpyOnlyUnpickler(BytesIO(raw)).load()
                assert set(planner) == {"left_joint_path", "right_joint_path"}
                counts = [
                    len(planner[key]) for key in ("left_joint_path", "right_joint_path")
                ]
                assert counts == [7, 4]
                for paths in planner.values():
                    for path in paths:
                        assert path["status"] == "Success"
                        assert path["position"].shape == path["velocity"].shape
                        assert path["position"].shape[1] == 6
                with h5py.File(sample_dir / f"arx-x5/episode{ep}.hdf5") as data:
                    recorded = data["joint_action/vector"][:]
                predicted, ticks, _ = timeline(planner, recorded[0])
                np.testing.assert_array_equal(recorded, predicted)
                spacings, frequencies = np.unique(np.diff(ticks), return_counts=True)
                episodes.append(
                    dict(
                        episode=ep,
                        member=member.filename,
                        sha256=sha256(raw).hexdigest(),
                        zip_crc32=f"{member.CRC:08x}",
                        bytes=len(raw),
                        planner_counts=counts,
                        recorded_frames=len(recorded),
                        reconstructed_frames=len(predicted),
                        max_target_abs_error=float(
                            np.max(np.abs(recorded - predicted))
                        ),
                        physics_ticks=int(ticks[-1]),
                        tick_spacing_counts={
                            str(k): int(v)
                            for k, v in zip(spacings, frequencies, strict=True)
                        },
                    )
                )
    prior_inventory = json.loads(
        (ROOT / AUDIT / "official-zip-directory.json").read_bytes()
    )
    for old, new in zip(prior_inventory, inventory, strict=True):
        assert all(old[k] == new[k] for k in ("archive", "hdf5", "planner_pkl_count"))
    prior_episodes = json.loads((ROOT / AUDIT / "arx-cache-replay.json").read_bytes())[
        "episodes"
    ]
    assert episodes == prior_episodes


def audit_upstream(ev, source_dir):
    lock = ev.read(AUDIT + SOURCES)
    for item in lock["files"]:
        rel = item["file"]
        ev.path(rel, source_dir)
        if ev.hashes[rel] != item["sha256"]:
            raise ValueError(f"upstream bytes changed: {rel}")
    manifest = ev.read("hf-manifest.json", source_dir)
    task = next(row for row in manifest["models"] if row["task"] == "handover_block")
    frozen = ev.read(
        G2 + "failure-audit-127054052-r1/release-context/pace_download_manifest.json"
    )
    stats = ev.read("hf-norm_stats.json", source_dir)["norm_stats"]
    files = {
        row["path"]: (row["bytes"], row["sha256"])
        for row in task["files"] + [manifest["tokenizer"]]
    }
    assert files == {
        row["path"]: (row["bytes"], row["sha256"]) for row in frozen["files"]
    }
    assert (
        ev.hashes["hf-norm_stats.json"]
        == task["norm_stats_sha256"]
        == frozen["norm_stats_sha256"]
    )
    assert all(
        len(values) == 14 for block in stats.values() for values in block.values()
    )
    base = G2 + "failure-audit-127054052-r1/release-context/"
    runtime_hashes = ev.read(base + "source-audit.json")
    matched = {
        name: ev.hashes["pace/" + name] == digest
        for name, digest in runtime_hashes.items()
        if "pace/" + name in ev.hashes
    }
    assert all(
        value
        for key, value in matched.items()
        if key != "task_config/_embodiment_config.yml"
    )
    selector = ev.read("pace/pace/configs/calibration/handover_block.json", source_dir)
    cache = ev.read("pace/pace/configs/expert_scenes.json", source_dir)["tasks"][
        "handover_block"
    ]
    datasets = {}
    for task_name in ("handover_block", "adjust_bottle"):
        files_for_task = ev.read(f"hf-data-{task_name}.json", source_dir)
        datasets[task_name] = {
            Path(row["path"]).name: {
                "bytes": row["size"],
                "lfs": row.get("lfs", {}).get("oid"),
            }
            for row in files_for_task
            if row["path"].endswith(
                ("aloha-agilex_clean_50.zip", "arx-x5_clean_50.zip", "demo_clean.zip")
            )
        }
    comparison = {
        item["file"][6:]: item["sha256"] == ev.hashes.get("runtime/" + item["file"][6:])
        for item in lock["files"]
        if item["file"].startswith("train/")
        and "runtime/" + item["file"][6:] in ev.hashes
    }
    spec = importlib.util.spec_from_file_location(
        "audited_pace_core", source_dir / "pace/pace/src/pace/core.py"
    )
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    config = core.PaceConfig(**selector["selector"])
    info = dict(
        norm_stats_sha256=frozen["norm_stats_sha256"],
        frozen_model_files=len(files),
        training_data_split=manifest["training_data_split"],
        demonstrations_per_task=manifest["demonstrations_per_task"],
        training_seed=manifest["training_seed"],
        checkpoint_step=manifest["checkpoint_step"],
        norm_schema={key: sorted(values) for key, values in stats.items()},
        seed_cache={
            seed: dict(count=len(rows), first_scene_seed=rows[0]["scene_seed"])
            for seed, rows in cache.items()
        },
        calibration=selector["selector"],
        calibration_threshold=selector["threshold"],
        runtime_source_sha_match=matched,
        train_runtime_source_byte_match=comparison,
        published_dataset_members=datasets,
        adjust_bottle_pinned_model_paths=[
            row["path"] for row in ev.read("hf-model-adjust_bottle.json", source_dir)
        ],
    )
    return info, lambda chunk: core.select_horizon(
        chunk, selector["threshold"], config=config
    )


def audit_native(ev, select_horizon):
    old_base = G2 + "failure-audit-127054052-r1/"
    old_actions = ev.read(old_base + "action-audit.json")
    old_trials = ev.read(old_base + "trial-audit.json")
    old = {}
    for trial, blocks in old_actions.items():
        if "-A-native-" not in trial:
            continue
        prefixes = [np.asarray(block["executed_prefix"]) for block in blocks]
        old[trial] = dict(
            queries=len(blocks),
            completed_actions=old_trials[trial]["result"]["executed"],
            success=old_trials[trial]["result"]["success"],
            tcp_rmse_mm=1000 * old_trials[trial]["tracking_position_rmse_m"],
            hands={
                hand: {
                    **close_counts(
                        [block["first_close_index"][i] for block in blocks],
                        [len(p) for p in prefixes],
                    ),
                    "first_closed_prefix_action": first_closed(
                        np.concatenate(prefixes), col
                    ),
                }
                for i, (hand, col) in enumerate(HANDS.items())
            },
        )
    base = G2 + "run-127081074/"
    events = ev.read(base + "derived/event-summary.json")
    terminal = ev.read(G2 + "final-cpu-review/failure-evidence.json")
    fcl = ev.read(G2 + "final-cpu-review/native-fcl-replay.json")
    trials = {}
    first_chunks = {}
    with zipfile.ZipFile(ev.path(base + "analysis-evidence.zip")) as archive:
        for trial, event in sorted(events.items()):
            requests = ev.read(f"{base}derived/{trial}/requests.json")
            result = ev.read(f"{base}trials/{trial}/result.json")
            k = result["window"]
            raw, selected = [], []
            for query in range(1, result["queries"] + 1):
                for kind, group in (("action", raw), ("plan", selected)):
                    name = f"trials/{trial}/{kind}-{query:05}.npy"
                    group.append(
                        np.load(BytesIO(archive.read(name)), allow_pickle=False)
                    )
            assert all(
                chunk.shape == (50, 14) and np.isfinite(chunk).all() for chunk in raw
            )
            assert all(len(p) <= k for p in selected)
            lengths = [len(p) for p in selected]
            horizons = [select_horizon(chunk) for chunk in raw]
            first_chunks[trial] = raw[0]
            hands = {}
            for hand, column in HANDS.items():
                indices = [first_closed(chunk, column) for chunk in raw]
                archived_indices = terminal["trials"][trial]["hands"][hand][
                    "first_raw_indices"
                ]
                assert indices == archived_indices
                command_ids = [
                    q["action_index_in_episode"]
                    for q in requests
                    if q["normalized_qpos_request"][column] < THRESHOLD
                ]
                hands[hand] = dict(
                    **close_counts(indices, lengths),
                    fixed25=close_counts(indices, [25] * len(raw)),
                    fixed50=close_counts(indices, [50] * len(raw)),
                    pace=close_counts(indices, horizons),
                    closure_requests=command_ids,
                    completed_closures=[
                        index for index in command_ids if index < result["executed"]
                    ],
                )
            trials[trial] = dict(
                window=k,
                queries=len(raw),
                completed_actions=result["executed"],
                physics_steps=event["physics_steps"],
                success=result["success"],
                first_holder_ticks=event["first_holder_ticks"],
                tcp_max_mm=1000 * max(event["tcp_position_max_m"]),
                pace_min=int(min(horizons)),
                pace_median=float(np.median(horizons)),
                pace_max=int(max(horizons)),
                hands=hands,
            )
            if trial in fcl:
                row = fcl[trial]
                trials[trial]["fcl_measured_start_clear"] = row["measured_clear"]
                trials[trial]["fcl_drive_reconstruction_error_rad"] = row[
                    "recorded_drive_max_error_rad"
                ]
                if "depth_m" in row["historical_rejection"]:
                    trials[trial]["historical_depth_mm"] = (
                        1000 * row["historical_rejection"]["depth_m"]
                    )
    with ev.path(G2 + "final-cpu-review/k26-slip.csv").open() as stream:
        slip = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    assert [int(row["tick"]) for row in slip] == list(range(4802, 4846))
    recent = ev.read(G2 + "final-cpu-review/result-127139405.json")
    return {
        "127054052": old,
        "127081074": trials,
        "127081074_first_chunk_max_abs_difference_by_seed": {
            str(seed): max(
                float(np.max(np.abs(first_chunks[f"g2e-seed{seed}-native-w1"] - chunk)))
                for trial, chunk in first_chunks.items()
                if f"seed{seed}-" in trial
            )
            for seed in (0, 1)
        },
        "slip_k26": dict(
            count=len(slip),
            first=slip[0],
            final=slip[-1],
            support_nonzero=sum(row["table_impulse_ns"] > 0 for row in slip),
        ),
        "release_k31": terminal["k31"],
        "127139405": dict(
            outcome=recent["outcome"],
            stop_trigger=recent["stop_trigger"],
            trials=[
                {
                    key: row[key]
                    for key in (
                        "episode_id",
                        "status",
                        "outcome",
                        "task_predicate",
                        "failure",
                        "policy_actions",
                        "recovery_actions",
                        "evidence",
                    )
                }
                for row in recent["trials"]
                if row["status"] == "completed"
            ],
        ),
    }


def audit_experts(ev, sample_dir, include_ik_details):
    workspace = G2 + "expert-workspace-20261106-r2/"
    ik = ev.read(workspace + "summary.json")["episodes"]
    path_checks = ev.read(
        G2 + "expert-path-collision-20261106-r2/local-verification.json"
    )["summary"]
    boundary = ev.read(G2 + "remaining-cpu-20261106-r1/expert-audit-r2.json")
    suffix = ev.read(G2 + "remaining-cpu-20261106-r1/placement-r2.json")
    native = ev.read(G2 + "native-expert-acceptance-20261106/acceptance.json")[
        "episodes"
    ]
    reference = ev.read(G2 + "arx-centered-20261106/acceptance.json")["episodes"]
    replay = ev.read(
        G2 + "offline-repair-20261106-r1/static-replay-127054052-r3/replay.json"
    )
    recovery = ev.read(G2 + "remaining-cpu-20261106-r1/replay-final.json")
    rows = []
    for ik_row, path, aloha, arx in zip(
        ik, path_checks, native, reference, strict=True
    ):
        ep = ik_row["episode"]
        row = dict(
            episode=ep,
            native_frames=ik_row["frames"],
            both_ik_solved=ik_row["both_ik_solved"],
            native_expert_success=aloha["native_task_success"],
            native_ee_vs_archived_mm=aloha["archived_ee_max_position_mm"],
            native_ee_vs_archived_deg=aloha["archived_ee_max_angle_deg"],
            blocked_static_segments=path["blocked_segments"],
            static_samples=path["samples"],
            blocked_measured_frame_boundaries=boundary[str(ep)]["rejected"],
            zero_step_suffix_accepted=suffix[str(ep)]["accepted"],
            retimed_expert_task_success=arx["task_success"],
            retimed_expert_hold_success=arx["hold_success"],
            retimed_expert_tool_tracking_mm=arx["max_actual_tool_position_error_mm"],
            retimed_expert_tool_tracking_deg=arx["max_actual_tool_angle_error_deg"],
            production_gate_exercised=arx["production_collision_gate_exercised"],
        )
        if include_ik_details:
            name = f"{workspace}episode{ep}.json"
            detail = ev.read(name)
            expected = ev.read(workspace + "checksums.json")
            assert (
                expected[name] == ev.hashes[name]
                and len(detail["records"]) == row["native_frames"]
            )
            tools = [
                arm["tcp_residual"]
                for record in detail["records"]
                for arm in record["arms"]
            ]
            row["arx_ik_fk_max_mm"] = 1000 * max(t["position_m"] for t in tools)
            row["arx_ik_fk_max_deg"] = max(t["angle_deg"] for t in tools)
        rows.append(row)
    manifest = ev.read("research/robotwin/sample_manifest.json")
    inventory = ev.read(AUDIT + "official-zip-directory.json")
    directories = {Path(row["archive"]).name: row for row in inventory}
    demo = directories["demo_clean.zip"]
    signatures = {(row["bytes"], row["crc32"]) for row in demo["hdf5"]}
    archive_comparison = dict(
        demo_clean_hdf5_dirs=sorted(
            {str(Path(row["member"]).parent) for row in demo["hdf5"]}
        ),
        demo_clean_hdf5_count=len(demo["hdf5"]),
        demo_clean_hdf5_matches_by_size_and_crc={
            archive: sum(
                (row["bytes"], row["crc32"]) in signatures for row in receipt["hdf5"]
            )
            for archive, receipt in directories.items()
            if archive != "demo_clean.zip"
        },
        planner_pkl_counts={
            archive: receipt["planner_pkl_count"]
            for archive, receipt in directories.items()
        },
    )
    arx_cache = ev.read(AUDIT + "arx-cache-replay.json")["episodes"]
    if sample_dir is not None:
        import h5py

        samples = []
        for archive in manifest["archives"]:
            for member in archive["selected"]:
                name = member["file"]
                if not name.endswith(".hdf5"):
                    continue
                data = ev.path(name, sample_dir)
                assert ev.hashes[name] == member["sha256"]
                with h5py.File(data, "r") as stream:
                    raw = stream["joint_action/vector"][:]
                    assembled = np.column_stack(
                        [
                            stream["joint_action/" + key][:]
                            for key in (
                                "left_arm",
                                "left_gripper",
                                "right_arm",
                                "right_gripper",
                            )
                        ]
                    )
                    assert np.array_equal(raw, assembled)
                    samples.append(
                        dict(
                            file=name,
                            frames=len(raw),
                            raw_columns=raw.shape[1],
                            root_action_exists="action" in stream,
                            next_frame_labels=len(raw) - 1,
                        )
                    )
    else:
        samples = None
    return dict(
        episodes=rows,
        original_expert_hdf5=samples,
        archive_comparison=archive_comparison,
        arx_native_cache=arx_cache,
        historical_arx=[
            dict(
                trial=trial,
                world=row["world"],
                original_status=row["original"]["status"],
                expanded_pairs=row["expanded_search"]["checked_pairs"],
                expert_endpoint_path_controls=[
                    item["result"]["accepted"]
                    for item in row["expert_task_space_controls"]
                ],
                exact_refinement_accepted=recovery[trial]["exact_refinement"][
                    "accepted"
                ],
                bounded_recovery_accepted=recovery[trial]["recovery"]["accepted"],
            )
            for trial, row in replay.items()
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--fetch-upstream", action="store_true")
    parser.add_argument("--recheck-official-zips", action="store_true")
    parser.add_argument(
        "--sample-dir", type=Path, help="optional six pinned HDF5 members"
    )
    parser.add_argument(
        "--expert-details",
        action="store_true",
        help="include private verified IK frame summaries",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.recheck_official_zips and not args.sample_dir:
        parser.error("--recheck-official-zips requires --sample-dir")
    if args.fetch_upstream:
        fetch_upstream(args.upstream_dir)
    ev = Evidence()
    upstream, selector = audit_upstream(ev, args.upstream_dir)
    native = audit_native(ev, selector)
    experts = audit_experts(ev, args.sample_dir, args.expert_details)
    if args.recheck_official_zips:
        official_zip_recheck(args.upstream_dir, args.sample_dir)
    for name in (
        "scripts/pace_runtime.py",
        "scripts/pace_model_smoke.py",
        "scripts/pace_client_smoke.py",
        "scripts/g2e_model_entry.py",
        "scripts/g2_capture.py",
        "scripts/g2_native_backend.py",
        "scripts/g2_episode.py",
        "scripts/g2e_native_guard.py",
        "scripts/g2e_protocol.py",
        "scripts/g2_task_runtime.py",
        "scripts/g2_adapter.py",
        "scripts/g2_assets.py",
        "scripts/g2_expert_dense.py",
        "scripts/g2_audit_expert_metadata.py",
        "scripts/g2_motion.py",
        "scripts/g2_arx_reference.py",
        "scripts/g2_remaining_coordinator.py",
        "scripts/g2_remaining_geometry.py",
        "scripts/sample_robotwin_archives.py",
        "scripts/pi05_robotwin_audit.py",
        G2 + "run-127054052/summary.json",
        G2 + "g2e-20261106-r1/release-hashes.json",
        G2 + "final-cpu-review/submission-127139405.json",
        G2 + "expert-metadata-20261106/audit.json",
        G2 + "expert-cache-uncached-20261106/provenance.json",
        "research/robotwin/embodiment_config_audit.json",
    ):
        ev.path(name)
    with args.output.open("x") as output:
        json.dump(
            dict(
                upstream=upstream,
                native=native,
                experts=experts,
                sha256=ev.hashes,
                model_loaded=False,
                gpu_used=False,
            ),
            output,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
