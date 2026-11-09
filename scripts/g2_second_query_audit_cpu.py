"""Audit saved policy exchanges/FK against pinned source and expert; no inference."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np

from g2_assets import load_arms
from g2_initial import digest_json, matrix7
from g2_kinematics import pose_error
from g2_native_expert_cpu import unpack_trace
from g2_unified_exchange import exchange_receipt, hash_array


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def pose_difference(a, b):
    e = pose_error(a, b)
    return dict(
        position_mm=float(np.linalg.norm(e[:3]) * 1000),
        angle_deg=float(np.rad2deg(np.linalg.norm(e[3:]))),
    )


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    ep = a.raw / "trials/unified-seed0-native-w50/episode"
    source = read(project / "research/g2/protocol-audit/source-lock.json")
    pinned = {}
    for row in source["files"]:
        p = a.protocol / row["file"]
        if p.is_file():
            assert digest(p) == row["sha256"], row["file"]
            pinned[row["file"]] = dict(sha256=row["sha256"], url=row["url"])
    config_path = a.release / "configs/unified-seed0-native-w50.json"
    config = read(config_path)
    initial = read(ep / "initial-readback.json")
    setup = read(ep / "setup.json")
    assert digest_json(config) == setup["config_sha256"]
    arms = load_arms(
        a.assets,
        a.release / "asset-audit.json",
        "aloha-agilex",
        config["calibration"]["aloha-agilex"],
    )
    frames = [
        np.linalg.inv(matrix7(initial["task_frame_world"]))
        @ matrix7(initial["robot_base_world"][s])
        for s in ("left", "right")
    ]

    def fk(row):
        row = np.asarray(row).reshape(2, 7)
        return np.array(
            [
                f @ arm.chain.fk(v[:6]) @ arm.offset()
                for f, arm, v in zip(frames, arms, row)
            ]
        )

    selected = read(a.raw / "selected-events.json")
    events = selected["selected"]

    def find(kind):
        return [r for r in events if r["event"].get("kind") == kind]

    snapshots = find("snapshot")
    inputs = find("policy_input")
    requests = find("native_action_request")
    projection = find("native_approach_projection")[0]
    barrier = find("native_approach_barrier")[0]
    last = selected["last"]["executed_step"]["event"]
    state2 = np.array(inputs[1]["event"]["state"], dtype=np.float32)
    assert len(snapshots) == len(inputs) == 2 and len(requests) == 28
    assert (
        snapshots[1]["event"]["stamp"]["tick"]
        == barrier["event"]["receipt"]["tick"]
        == last["tick"]
        == 3575
    )
    assert (
        inputs[0]["event"]["prompt"] == inputs[1]["event"]["prompt"] == config["prompt"]
    )
    assert np.array_equal(
        state2.reshape(2, 7)[:, :6],
        np.asarray(
            projection["event"]["approved_qpos_request"], dtype=np.float32
        ).reshape(2, 7)[:, :6],
    )
    norm_path = a.protocol / "hf-norm_stats.json"
    norms = read(norm_path)["norm_stats"]
    startup = read(a.raw / "model/startup.json")
    manifest = read(a.release / "pace_download_manifest.json")
    assert (
        digest(norm_path)
        == startup["checkpoint_files"][manifest["norm_stats"]][2]
        == manifest["norm_stats_sha256"]
    )
    assert startup["checkpoint_revision"] == source["revisions"]["checkpoint"]
    assert all(
        startup["checkpoint_files"][r["path"]][0] == r["bytes"]
        and startup["checkpoint_files"][r["path"]][2] == r["sha256"]
        for r in manifest["files"]
    )
    instructions = read(
        a.protocol / "pace/description/task_instruction/handover_block.json"
    )
    prompt = config["prompt"]
    chunks, cameras = [], []
    states = []
    raw_actions = []
    arm_columns = [*range(6), *range(7, 13)]
    for i in (1, 2):
        raw = np.load(ep / f"action-{i:05}.npy", allow_pickle=False)
        plan = np.load(ep / f"plan-{i:05}.npy", allow_pickle=False)
        images = np.load(ep / f"observation-{i:05}.npz", allow_pickle=False)
        state = np.array(inputs[i - 1]["event"]["state"], dtype=np.float32)
        assert raw.shape == plan.shape == (50, 14) and np.isfinite(raw).all()
        expected = raw.copy()
        expected[:, [6, 13]] = np.clip(raw[:, [6, 13]], 0, 1)
        assert np.array_equal(expected, plan) and np.array_equal(
            raw[:, arm_columns], plan[:, arm_columns]
        )
        cm = {}
        for name in images.files:
            rgb = images[name]
            assert rgb.shape == (3, 240, 320) and rgb.dtype == np.uint8
            # Descriptive color counts, not segmentation or a visibility proof.
            cm[name] = dict(
                shape=list(rgb.shape),
                dtype=str(rgb.dtype),
                array_sha256=hash_array(rgb),
                mean_channels=rgb.mean(axis=(1, 2)).tolist(),
                red_dominant_pixels=int(
                    (
                        (rgb[0].astype(int) > rgb[1].astype(int) + 50)
                        & (rgb[0].astype(int) > rgb[2].astype(int) + 50)
                    ).sum()
                ),
            )
        cameras.append(cm)
        stats = norms["state"]
        normalized = (state - stats["q01"]) / (
            np.array(stats["q99"]) - stats["q01"] + 1e-6
        ) * 2 - 1
        states.append(
            dict(
                state=state.tolist(),
                normalized_using_matching_stats=normalized.tolist(),
                max_abs_normalized=float(abs(normalized).max()),
                outside_q01_q99=np.flatnonzero(abs(normalized) > 1).tolist(),
            )
        )

        def close(column):
            v = np.flatnonzero(raw[:, column] < 0.5)
            return int(v[0]) if len(v) else None

        chunks.append(
            dict(
                query=i,
                raw_file_sha256=digest(ep / f"action-{i:05}.npy"),
                plan_file_sha256=digest(ep / f"plan-{i:05}.npy"),
                raw_array_sha256=hash_array(raw),
                shape=list(raw.shape),
                dtype=str(raw.dtype),
                first_row=raw[0].tolist(),
                first_tcp_task=fk(raw[0]).tolist(),
                final_tcp_task=fk(raw[-1]).tolist(),
                first_left_close_index_zero_based=close(6),
                first_right_close_index_zero_based=close(13),
                clipped_gripper_values=int(
                    np.count_nonzero(raw[:, [6, 13]] != plan[:, [6, 13]])
                ),
                first_row_directly_matches_raw=bool(np.array_equal(raw[0], plan[0])),
                arm_targets_byte_equal_to_raw=True,
            )
        )
        raw_actions.append(raw)
        if i == 1:
            exchange = exchange_receipt(
                dict(
                    state=state,
                    images={k: images[k] for k in images.files},
                    prompt=prompt,
                ),
                dict(actions=raw),
            )
            assert (
                exchange
                == read(a.raw / "trials/unified-seed0-native-w50/first-exchange.json")[
                    "observed"
                ]["first_exchange"]
            )
    for row, request in zip(raw_actions[0][:27], requests[:27]):
        expect = row.copy()
        expect[[6, 13]] = np.clip(expect[[6, 13]], 0, 1)
        np.testing.assert_array_equal(
            expect, request["event"]["normalized_qpos_request"]
        )
    np.testing.assert_array_equal(
        np.load(ep / "plan-00002.npy")[0],
        requests[-1]["event"]["normalized_qpos_request"],
    )
    all_tcp = np.asarray([[fk(row) for row in block] for block in raw_actions])
    tcp_file = a.output / "predicted-tcp.npz"
    np.savez_compressed(
        tcp_file,
        tcp_task=all_tcp,
        raw_gripper_requests=np.asarray(raw_actions)[:, :, [6, 13]],
    )
    assert requests[-1]["event"]["tick"] == 3575
    state2_fk = fk(state2)
    raw2_fk = fk(raw_actions[1][0])
    nested = read(ep / "result.json")["failure"]["diagnostic"]["attempts"][0][
        "diagnostic"
    ]["collision_examples"][0]
    refined_row = raw_actions[1][0].reshape(2, 7).copy()
    refined_row[:, :6] = nested["candidate_q"]
    refined_fk = fk(refined_row)
    intended_refinement = raw2_fk.copy()
    intended_refinement[0, 2, 3] += 0.0115
    # Standardize expert data to the same TCP convention. The recorded expert
    # joint-frame endpose needs translation and a fixed right rotation; it is
    # not itself the policy TCP. Verify the convention on all approach samples.
    expert_meta = read(
        project / "research/g2/expert-dense-episode0-20261106/input.json"
    )
    trace_path = (
        project / "research/g2/expert-dense-episode0-20261106/runtime-trace.npz"
    )
    expert = unpack_trace(trace_path)
    physical_path = (
        project / "research/g2/native-expert-episode0-20261106/physics-trace.npz"
    )
    measured = np.load(physical_path, allow_pickle=False)
    measured_initial = read(
        project / "research/g2/native-expert-episode0-20261106/initial.json"
    )
    assert (
        np.max(
            abs(
                np.array(measured_initial["block_pose"])[:3]
                - initial["objects_task"]["block"][:3]
            )
        )
        == 0
    )
    pregrasp_end = expert_meta["segments"][1]["last_tick"]
    pregrasp = []
    convention_errors = []
    for tick in range(pregrasp_end + 1):
        qa = measured["actual_native"][tick].reshape(2, 8)
        row = np.c_[qa[:, :6], measured["normalized_gripper_target"][tick]].ravel()
        tool = fk(row)
        endpose = measured["actual_endpose_matrix"][tick].copy()
        endpose[:, :3, 3] += 0.12 * endpose[:, :3, 0]
        endpose[:, :3, :3] = endpose[:, :3, :3] @ np.diag([1.0, -1.0, -1.0])
        convention_errors.append(float(np.max(abs(tool - endpose))))
        pregrasp.append((tick, tool, row))
    nearest = min(
        pregrasp,
        key=lambda x: np.linalg.norm(
            x[1][0, :3, 3] - np.array(last["actual_tools_task"])[0, :3, 3]
        ),
    )
    phase_refs = []
    for tick in (576, 709):
        actual_ref = pregrasp[tick][1]
        phase_refs.append(
            dict(
                expert_tick=tick,
                model_state_time_aligned=False,
                expert_command_tcp=fk(expert["target"][tick])[0].tolist(),
                expert_measured_tcp=actual_ref[0].tolist(),
                expert_gripper=expert["target"][tick, [6, 13]].tolist(),
                reference_only_difference=pose_difference(
                    np.array(last["actual_tools_task"])[0], actual_ref[0]
                ),
            )
        )
    nearest_error = pose_difference(
        np.array(last["actual_tools_task"])[0], nearest[1][0]
    )
    assert max(convention_errors) < 2e-6
    report = dict(
        job="127192740",
        audit_base="0467a96258a2e3982196ae928a153ed85bddce8d",
        gpu_used=False,
        model_inference=False,
        physics_steps=0,
        runtime_modified=False,
        model_files_match_frozen_manifest=True,
        norm_stats_sha256=digest(norm_path),
        config_sha256=digest(config_path),
        config_canonical_sha256=setup["config_sha256"],
        pinned_revisions=source["revisions"],
        pinned_source_files=pinned,
        input=dict(
            camera_stats=cameras,
            states=states,
            prompt=prompt,
            prompt_in_seen=prompt in instructions["seen"],
            prompt_in_unseen=prompt in instructions["unseen"],
            first_exchange_independently_reproduced=exchange,
            second_server_input_echo_available=False,
            snapshot_lines=[r["source_line"] for r in snapshots],
            policy_input_lines=[r["source_line"] for r in inputs],
            second_tick=3575,
            barrier_tick=3575,
            discarded_old_tail=23,
            second_state_tracks_approved_arm_float32_exactly=True,
            source_state_semantics="drive targets + requested normalized grippers, not actual arm joints",
            normalized_values_are_distribution_indicators_not_training_membership_tests=True,
        ),
        output=dict(
            chunks=chunks,
            all_100_predicted_tcp_file="predicted-tcp.npz",
            all_100_predicted_tcp_sha256=digest(tcp_file),
            all_100_predicted_tcp_shape=list(all_tcp.shape),
            two_prediction_arrays_identical=bool(np.array_equal(*raw_actions)),
            approved_prefix_requests_match_logged_requests=True,
            second_raw_tcp_delta_from_query_state_mm=(
                (raw2_fk[0, :3, 3] - state2_fk[0, :3, 3]) * 1000
            ).tolist(),
            second_raw_rotation_delta_from_state=pose_difference(
                raw2_fk[0], state2_fk[0]
            ),
            second_state_vs_actual_tcp=pose_difference(
                state2_fk[0], np.array(last["actual_tools_task"])[0]
            ),
            second_raw_tcp=raw2_fk[0].tolist(),
            logged_corrected_candidate_tcp=refined_fk[0].tolist(),
            corrected_vs_raw_mm=(
                (refined_fk[0, :3, 3] - raw2_fk[0, :3, 3]) * 1000
            ).tolist(),
            corrected_refinement_residual=pose_difference(
                intended_refinement[0], refined_fk[0]
            ),
            final_rejection_refers_to="plus-z refinement candidate, not the untouched raw goal",
            logged_corrected_depth_m=nested["collisions"][0]["depth_m"],
            second_actions_executed=0,
        ),
        expert=dict(
            source_trace_sha256=digest(trace_path),
            measured_trace_sha256=digest(physical_path),
            recorded_joint_frame_to_tcp_offset_m=0.12,
            recorded_joint_frame_to_tcp_right_rotation=np.diag(
                [1.0, -1.0, -1.0]
            ).tolist(),
            maximum_convention_conversion_element_error=max(convention_errors),
            settled_block_position_exact_match=True,
            pregrasp_references=phase_refs,
            nearest_pregrasp_measured_position_tick=nearest[0],
            nearest_full_pose_difference=nearest_error,
            exact_matched_expert_state_available=False,
            warning="No time-index action subtraction or expert next-action label at the policy state; endpoint comparison is a phase reference only.",
        ),
        raw_manifest=read(a.raw / "raw-manifest.json"),
        source_script_sha256=digest(Path(__file__)),
    )
    (a.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    # Small public evidence: predictions and key numeric logs; images and dense
    # expert data remain in the original private evidence archive.
    with zipfile.ZipFile(a.output / "key-evidence.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for i in (1, 2):
            for prefix in ("action", "plan"):
                name = f"{prefix}-{i:05}.npy"
                z.write(ep / name, name)
        keys = {
            "snapshot",
            "policy_input",
            "native_approach_projection",
            "native_approach_barrier",
            "native_guard_rejection",
        }
        z.writestr(
            "numeric-events.json",
            json.dumps(
                dict(
                    selected=[r for r in events if r["event"].get("kind") in keys],
                    last=selected["last"],
                )
            ),
        )
    print(
        json.dumps(
            dict(
                first_exchange_verified=True,
                goals_verified=True,
                state2_descent_mm=report["output"][
                    "second_raw_tcp_delta_from_query_state_mm"
                ][2],
                expert_nearest=nearest_error,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "release", "assets", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
