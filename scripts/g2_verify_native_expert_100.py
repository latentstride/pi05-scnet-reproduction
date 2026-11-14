"""Validate per-episode native physics evidence and emit the acceptance ledger."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from g2_native_expert_batch import ROBOTS, REVISION, SOURCE, read, sha, write
from robotwin_control_geometry import pose_from_wxyz


def final_predicate(block, tray, right):
    bottom = pose_from_wxyz(block[:3], block[3:]) @ [0., 0., -.1, 1.]
    top = pose_from_wxyz(tray[:3], tray[3:]) @ [0., 0., .005, 1.]
    delta = (bottom - top)[:3]
    return bool(np.all(np.abs(delta[:2]) < .03) and abs(delta[2]) < .01 and right > .8), delta


def first_tick(condition):
    hits = np.flatnonzero(condition)
    return int(hits[0]) if len(hits) else None


def dense_contacts(trace):
    mask = trace["contacts"][:, 0].astype(int)
    height = trace["block_pose"][:, 2]
    shared = mask & 15 == 15
    return dict(
        left_bilateral_lift=first_tick((mask & 3 == 3) & (height - height[0] >= .02)),
        both_bilateral=first_tick(shared),
        right_bilateral_after_shared=first_tick(np.logical_or.accumulate(shared) & (mask & 15 == 12)),
        target_contact=first_tick(mask & 16 != 0),
    )


def verify_episode(item, robot, episode, prepared=None, source_manifest=None):
    def require(condition, description):
        if not condition:
            raise ValueError(f"{robot} episode {episode}: {description}")

    receipt = read(item / "receipt.json")
    meta = receipt["input"]
    require((meta["robot"], meta["episode"], meta["dataset_revision"]) == (robot, episode, REVISION)
            and type(meta["episode"]) is int, "receipt identity")
    steps, status = receipt["physical_steps"], receipt["status"]
    if status == "NOT_REPRODUCIBLE":
        require(receipt["native_success"] is None and receipt.get("category") in (
            "RAW_DATA_MISSING_OR_CORRUPT", "CONTROL_CACHE_OR_TIMING",
            "SCENE_INITIALIZATION", "SIMULATION_OR_DYNAMICS",
        ) and bool(receipt.get("detail")), "unverified result requires a classified cause")
        return receipt, {}, {}

    require(status in ("SUCCESS", "TASK_FAILED") and type(receipt["native_success"]) is bool,
            "unknown result status or invalid predicate")
    require(receipt["category"] == (None if status == "SUCCESS" else "NORMAL_TASK_FAILURE"),
            "result category")
    require(meta["prepared"] and meta["source_revision"] == SOURCE and meta["exact_recorded_targets"]
            and type(steps) is int and steps == meta["steps"] == receipt["expected_steps"] > 0,
            "native control schedule completeness")
    for field in ("hdf5_sha256", "pkl_sha256", "seed_file_sha256", "controls_sha256",):
        require(re.fullmatch("[a-f0-9]{64}", meta[field]) is not None, f"invalid {field}")
    require(meta["hdf5_member"] == f"{robot}_clean_50/data/episode{episode}.hdf5"
            and meta["pkl_member"] == f"{robot}_clean_50/_traj_data/episode{episode}.pkl",
            "input member identity")

    reused = receipt.get("reused", False)
    if reused:
        require(robot == ROBOTS[0] and episode in (0, 1, 2)
                and receipt["control"] == "historical native Aloha dense drive replay"
                and receipt["legacy_receipt_sha256"] == sha(item / "attempt/replay.json"),
                "historical native evidence binding")
    else:
        require(receipt["control"] == "original native cached drives"
                and receipt["physics_system"] == "PhysxCpuSystem"
                and receipt["flags"] == dict(renderer=False, gpu=False, policy=False, cross_embodiment=False),
                "native CPU run identity")
        require(0 <= max(receipt["initialization"]["initial_ee_error_m"]) < .001
                and receipt["physics_config"]["initial_seed"] == meta["seed"], "measured initialization")
        if source_manifest:
            original = {e["path"]: e.get("sha256") for e in source_manifest["files"]}
            require(source_manifest["revision"] == SOURCE and len(receipt["source_hashes"]) == 8
                    and all(original[name] == digest for name, digest in receipt["source_hashes"].items()),
                    "official task and robot source digest")

    trace_path = item / "attempt/physics-trace.npz"
    require(sha(trace_path) == receipt["physics_trace_sha256"], "physics trace digest")
    with np.load(trace_path, allow_pickle=False) as trace:
        n = steps + 1 + (200 if reused else 0)
        actual_key = "actual_native" if reused else "actual_qpos"
        grip_key = "normalized_gripper_target" if reused else "normalized_gripper_request"
        drive_key = "drive_native" if reused else "native_drive"
        actual, grip, drive = trace[actual_key], trace[grip_key], trace[drive_key]
        ticks, blocks, physical_contact = trace["frame_ticks"], trace["block_pose"], trace["contacts"]
        actual_ee = trace["actual_endpose_matrix"]
        require(actual.shape == drive.shape == (n, 16) and grip.shape == (n, 2)
                and blocks.shape == (n, 7) and physical_contact.shape == (n, 8 if reused else 6)
                and actual_ee.shape == (n, 2, 4, 4)
                and all(np.isfinite(x).all() for x in (actual, grip, drive, blocks, physical_contact, actual_ee)),
                "physical trace shape or finite values")
        require(ticks[0] == 0 and ticks[-1] == steps and len(ticks) == meta["frames"]
                and np.all(np.diff(ticks) >= 0), "archived frame tick alignment")
        final = receipt["final"]
        actual_grip = grip[steps, 1]
        require(np.array_equal(blocks[steps], final["block_pose"])
                and np.array_equal(actual[steps], final["qpos"])
                and actual_grip == final["right_gripper_request"], "measured terminal state")
        initial = read(item / "attempt/initial.json")
        require(np.array_equal(initial["target_pose"], final["target_pose"]), "static target pose")
        if reused:
            contacts = historical_contacts(item / "attempt/contact-events.jsonl", steps)
            left, right = physical_contact[:steps + 1, :2].T > 0
        else:
            contacts = dense_contacts(trace)
            require(contacts == receipt["contacts"], "recomputed contact phases")
            left = np.any(physical_contact[:, 2:4] > 0, axis=1)
            right = np.any(physical_contact[:, 4:6] > 0, axis=1)
        shared = left & right
        phase = dict(contacts,
            left_contact_and_lift_tick=first_tick(left & (blocks[:steps + 1, 2] - blocks[0, 2] >= .02)),
            simultaneous_hand_contact_tick=first_tick(shared),
            right_only_after_shared_contact_tick=first_tick(np.logical_or.accumulate(shared) & right & ~left),
            left_contact_steps=int(left.sum()), right_contact_steps=int(right.sum()),
            both_hands_contact_steps=int(shared.sum()))

        if prepared:
            require(meta == read(prepared / "input.json"), "prepared input manifest mismatch")
            controls = prepared / "controls.npz"
            require(sha(controls) == meta["controls_sha256"], "reconstructed controls digest")
            with np.load(controls, allow_pickle=False) as source:
                expected = source["target"]
                require(np.array_equal(drive[:steps + 1, :6], expected[:, :6])
                        and np.array_equal(drive[:steps + 1, 8:14], expected[:, 7:13])
                        and np.array_equal(grip[:steps + 1], expected[:, [6, 13]])
                        and np.array_equal(ticks, source["frame_ticks"]), "executed drives differ from cache")
                if not reused:
                    target = np.array([[pose_from_wxyz(p[:3], p[3:]) for p in pair]
                                       for pair in source["expected_endpose"]])
                    replayed = actual_ee[ticks]
                    distance = np.linalg.norm(replayed[..., :3, 3] - target[..., :3, 3], axis=-1)
                    rotation = np.swapaxes(target[..., :3, :3], -1, -2) @ replayed[..., :3, :3]
                    angle = np.rad2deg(np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
                    error = receipt["ee_error"]
                    require(np.isclose(distance.max(), error["max_m"], rtol=0, atol=1e-9)
                            and np.isclose(angle.max(), error["max_deg"], rtol=0, atol=1e-6)
                            and (distance.max() < .001 and angle.max() < .1) == error["within_prior_tolerance"],
                            "archived observed endpose discrepancy")

    native, delta = final_predicate(final["block_pose"], final["target_pose"], actual_grip)
    require(native == receipt["native_success"] == (status == "SUCCESS")
            and np.allclose(delta, final["functional_delta_m"], atol=2e-7, rtol=0),
            "original check_success from terminal physics")
    require(receipt["physics_config"]["dt_s"] == .004
            and abs(receipt["physics_config"]["mass_kg"] - .01) < 1e-8,
            "physical timestep or block mass")
    return receipt, phase, {"controls_match": True if prepared else None}


def historical_contacts(events, last_tick):
    result = dict(left_bilateral_lift=None, both_bilateral=None,
                  right_bilateral_after_shared=None, target_contact=None)
    left, right = {"fl_link7", "fl_link8"}, {"fr_link7", "fr_link8"}
    base_height = None
    for line in events.read_text().splitlines():
        row = json.loads(line)
        tick, height = row["tick"], row["box_pose"][2]
        if tick > last_tick:
            break
        if base_height is None:
            base_height = height
        contacts = set()
        for hit in row["contacts"]:
            if "box" in hit["actors"] and any(any(point["impulse"]) for point in hit["points"]):
                contacts.update(hit["actors"])
        if left <= contacts and height - base_height >= .02 and result["left_bilateral_lift"] is None:
            result["left_bilateral_lift"] = tick
        if left | right <= contacts and result["both_bilateral"] is None:
            result["both_bilateral"] = tick
        if result["both_bilateral"] is not None and right <= contacts and not left & contacts:
            if result["right_bilateral_after_shared"] is None:
                result["right_bilateral_after_shared"] = tick
        if "target_box" in contacts and result["target_contact"] is None:
            result["target_contact"] = tick
    return result


def legacy_receipts(source, prepared, output):
    bases = source / "research/g2"
    verified = read(bases / "native-expert-acceptance-20261106/acceptance.json")
    folder0 = bases / "native-expert-episode0-20261106"
    rest = bases / "native-expert-rest-20261106"
    checks = read(folder0 / "checksums.json") | read(rest / "checksums.json")
    for episode in range(3):
        folder = folder0 if episode == 0 else rest / f"episode{episode}"
        meta = read(prepared / ROBOTS[0] / f"episode{episode:02}" / "input.json")
        replay = read(folder / "replay.json")
        init = read(folder / "initial.json")
        archive_input = read(folder / "input.json")
        for name in ("replay.json", "initial.json", "contact-events.jsonl", "physics-trace.npz", "input.json"):
            member = folder / name
            if checks[str(member.relative_to(source))] != sha(member):
                raise ValueError(f"legacy artifact hash differs: {member}")
        if meta["pkl_sha256"] != archive_input["planner_sha256"] or meta["seed"] != replay["seed"]:
            raise ValueError("historical native trace belongs to a different input")
        n = replay["physics_steps_replayed"]
        prior = verified["episodes"][episode]
        if prior["episode"] != episode or not prior["verification_passed"] or n != meta["steps"]:
            raise ValueError("legacy episode did not pass prior independent acceptance")
        with np.load(folder / "physics-trace.npz", allow_pickle=False) as trace:
            block = trace["block_pose"][n]
            requested = trace["normalized_gripper_target"][n, 1]
            q = trace["actual_native"][n]
        success, delta = final_predicate(block, init["target_pose"], requested)
        if not replay["completed"] or success != replay["native_success_at_expert_end"]["success"]:
            raise ValueError("legacy predicate not verified")
        out = output / ROBOTS[0] / f"episode{episode:02}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "receipt.json").exists():
            raise ValueError("legacy receipt would overwrite an existing run")
        import shutil
        attempt = out / "attempt"
        attempt.mkdir()
        for name in ("replay.json", "initial.json", "contact-events.jsonl", "physics-trace.npz", "stages.json"):
            shutil.copyfile(folder / name, attempt / name)
        receipt = dict(input=meta, status="SUCCESS" if success else "TASK_FAILED",
            category=None if success else "NORMAL_TASK_FAILURE", native_success=success,
            control="historical native Aloha dense drive replay", physical_steps=n, expected_steps=n,
            reused=True, legacy_receipt_sha256=sha(folder / "replay.json"),
            simulator=replay["sapien_version"], physics_trace_sha256=sha(folder / "physics-trace.npz"),
            physics_config=dict(dt_s=.004, mass_kg=replay["actual_block_mass_kg"]),
            contacts=historical_contacts(folder / "contact-events.jsonl", n),
            contact_scope="historical event stream sampled when contact-pair identities changed; blank is unverified",
            ee_error=dict(max_m=prior["archived_ee_max_position_mm"]/1000, max_deg=prior["archived_ee_max_angle_deg"],
                          within_prior_tolerance=True),
            final=dict(block_pose=block.tolist(), target_pose=init["target_pose"], qpos=q.tolist(),
                       right_gripper_request=float(requested), functional_delta_m=delta.tolist()))
        write(out / "receipt.json", receipt)


def results(directory, prepared=None, manifests=None, source_manifest=None, hdf5_manifests=()):
    expected = {(robot, episode) for robot in ROBOTS for episode in range(50)}
    actual = {(p.parent.parent.name, p.parent.name) for p in directory.glob("*/*/receipt.json")}
    if actual != {(robot, f"episode{episode:02}") for robot, episode in expected}:
        raise ValueError("results directory must contain exactly 50 unique receipts per robot")
    originals = {}
    for path in hdf5_manifests:
        source = read(path)
        if source["revision"] != REVISION:
            raise ValueError(f"wrong HDF5 dataset revision at {path}")
        for row in source["records"]:
            if row.get("crc_verified") and row.get("sha256"):
                pair = row["robot"], row["episode"]
                member = row["member"], row["sha256"]
                if pair in originals and originals[pair] != member:
                    raise ValueError(f"conflicting HDF5 archive members for {pair}")
                originals[pair] = member
    rows = []
    for robot, episode in sorted(expected, key=lambda pair: (ROBOTS.index(pair[0]), pair[1])):
        relative = Path(robot) / f"episode{episode:02}"
        item = directory / relative
        receipt, contacts, checks = verify_episode(
            item, robot, episode, prepared / relative if prepared else None, source_manifest)
        meta = receipt["input"]
        if hdf5_manifests and meta.get("prepared"):
            if originals.get((robot, episode)) != (meta["hdf5_member"], meta["hdf5_sha256"]):
                raise ValueError(f"wrong HDF5 archive member at {item}")
        if manifests and meta.get("prepared"):
            manifest = manifests / robot
            index, seeds = read(manifest / "manifest.json"), (manifest / "seed.txt").read_text().split()
            if (index["robot"], index["revision"], int(seeds[episode])) != (robot, REVISION, meta["seed"]):
                raise ValueError(f"wrong archive manifest or episode seed at {item}")
            member = index["members"][f"episode{episode}.pkl"]
            if (member["sha256"], member["member"]) != (meta["pkl_sha256"], meta["pkl_member"]):
                raise ValueError(f"wrong planner member at {item}")
            if sha(manifest / "seed.txt") != meta["seed_file_sha256"]:
                raise ValueError(f"wrong seed list at {item}")
        error = receipt.get("ee_error", {})
        rows.append(dict(robot=robot, episode=episode, seed=meta.get("seed"), status=receipt["status"],
            native_check_success=receipt["native_success"], physical_steps=receipt["physical_steps"],
            frames=meta.get("frames"),
            left_bilateral_lift_tick=contacts.get("left_bilateral_lift"),
            both_bilateral_tick=contacts.get("both_bilateral"),
            right_takeover_tick=contacts.get("right_bilateral_after_shared"),
            placement_contact_tick=contacts.get("target_contact"),
            left_contact_and_lift_tick=contacts.get("left_contact_and_lift_tick"),
            simultaneous_hand_contact_tick=contacts.get("simultaneous_hand_contact_tick"),
            right_only_after_shared_contact_tick=contacts.get("right_only_after_shared_contact_tick"),
            left_contact_steps=contacts.get("left_contact_steps"), right_contact_steps=contacts.get("right_contact_steps"),
            both_hands_contact_steps=contacts.get("both_hands_contact_steps"),
            max_recorded_ee_error_m=error.get("max_m"), max_recorded_ee_error_deg=error.get("max_deg"),
            within_prior_trace_tolerance=error.get("within_prior_tolerance"),
            category=receipt.get("category"), detail=receipt.get("detail"), reused=bool(receipt.get("reused")),
            hdf5_sha256=meta.get("hdf5_sha256"), pkl_sha256=meta.get("pkl_sha256"),
            trace_sha256=receipt.get("physics_trace_sha256"),
            receipt_path=str(relative / "receipt.json"), receipt_sha256=sha(item / "receipt.json"),
            reconstructed_controls_verified=checks.get("controls_match")))
    for key in ("hdf5_sha256", "pkl_sha256"):
        values = [row[key] for row in rows if row[key]]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {key}")
    seeds = [(row["robot"], row["seed"]) for row in rows if row["seed"] is not None]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed within the same embodiment")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--import-historical", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--manifests", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--hdf5-manifests", type=Path, nargs="+", default=[])
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--check-csv", action="store_true")
    args = parser.parse_args()
    if args.import_historical:
        if args.prepared is None:
            parser.error("--import-historical requires --prepared")
        legacy_receipts(args.import_historical, args.prepared, args.directory)
    if args.evidence_manifest:
        for member, expected_sha in read(args.evidence_manifest).items():
            parent = args.evidence_manifest.parent.resolve()
            file = (parent / member).resolve()
            if parent not in file.parents or sha(file) != expected_sha:
                raise ValueError(f"evidence digest mismatch: {member}")
    rows = results(args.directory, args.prepared, args.manifests,
                   read(args.source_manifest) if args.source_manifest else None, args.hdf5_manifests)
    if args.check_csv:
        with args.csv_path.open(newline="") as stream:
            expected_rows = list(csv.DictReader(stream))
        actual_rows = [{k: str(v) if v is not None else "" for k, v in row.items()} for row in rows]
        # The standalone verification does not need the prepared binary inputs.
        if args.prepared is None:
            for row in expected_rows:
                row["reconstructed_controls_verified"] = ""
        if actual_rows != expected_rows:
            raise ValueError("CSV contents differ from verified evidence")
    else:
        with args.csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    for name in ROBOTS:
        selected = [row for row in rows if row["robot"] == name]
        print(name, {status: sum(r["status"] == status for r in selected)
                     for status in ("SUCCESS", "TASK_FAILED", "NOT_REPRODUCIBLE")})
