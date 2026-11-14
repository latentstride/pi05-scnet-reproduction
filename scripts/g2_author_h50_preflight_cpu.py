"""Inspect immutable G1 camera evidence and exercise the tick-zero stop boundary."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np

from g2_adapter import G2Error
from g2_final_trial import RuntimeReset
from g2_native_backend import SceneClock
from robotwin_control_geometry import Status
from test_g2_candidate_stop import fixture


ROOT = Path(__file__).resolve().parents[1]


def archived_first_frame():
    archive = ROOT / "docs/pace_g1_126905534_evidence.tar.gz"
    prefix = "pace-g1-126905534/g1_3"
    with tarfile.open(archive) as stream:
        summary = json.load(stream.extractfile(f"{prefix}/summary.json"))
        rgb_bytes = stream.extractfile(f"{prefix}/initial_observation.npz").read()
    assert summary["scene_seed"] == 100000
    assert summary["cache_split_seed"] == summary["cache_index"] == 0
    cameras = {}
    with np.load(BytesIO(rgb_bytes), allow_pickle=False) as data:
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            pixels = data[name]
            assert pixels.shape == (3, 240, 320) and pixels.dtype == np.uint8
            assert pixels.min() < pixels.max()
            cameras[name] = dict(
                shape=list(pixels.shape),
                dtype=str(pixels.dtype),
                sha256=sha256(pixels.tobytes()).hexdigest(),
                min=int(pixels.min()),
                max=int(pixels.max()),
            )
        state = data["state"].tolist()
    return dict(
        source_job="126905534",
        archive_sha256=sha256(archive.read_bytes()).hexdigest(),
        original_npz_member=f"{prefix}/initial_observation.npz",
        original_npz_sha256=sha256(rgb_bytes).hexdigest(),
        scene_seed=summary["scene_seed"],
        cache_sha256=summary["scene_cache"]["sha256"],
        pace_revision=summary["pace_revision"],
        initial_poses=summary["initial_poses"],
        drive_state=state,
        cameras=cameras,
        prompt=summary["prompt"],
        control_steps=summary["control_steps"],
        scope="Archived real first frame of the G1 interface smoke; not a current live reset or a Fixed H50 result.",
    )


def spawn_rng_check(archive):
    """Recompute native spawn draws and check XY against historical settled poses."""
    import yaml

    plan = json.loads(
        (ROOT / "research/g2/author-h50-preflight/scene-config.json").read_text()
    )
    embodiment = plan["embodiment"]
    path = ROOT / "research/robotwin/embodiment_assets" / embodiment["config_member"]
    assert sha256(path.read_bytes()).hexdigest() == embodiment["config_sha256"]
    config = yaml.safe_load(path.read_text())
    rng = np.random.RandomState(archive["scene_seed"])
    rng.uniform(
        0, 0
    )  # _init_task_env_: table_z_bias, even with no height randomization.
    for _ in config["static_camera_list"]:
        rng.randn(3)  # Camera.load_camera: direction draw, even with zero displacement.
        rng.uniform(0, 0)
    block = [rng.uniform(-0.25, -0.05), rng.uniform(0, 0.25), rng.uniform(0.842, 0.842)]
    block_euler = [rng.uniform(-limit, limit) for limit in (0.0, 0.0, 0.2)]
    target = [rng.uniform(0.1, 0.25), rng.uniform(0.15, 0.2), rng.uniform(0.741, 0.741)]
    errors = {
        name: float(
            np.linalg.norm(
                np.asarray(pose[:2]) - archive["initial_poses"][name]["position"][:2]
            )
        )
        for name, pose in (("block", block), ("target", target))
    }
    assert errors["block"] < 0.001 and errors["target"] < 1e-6
    return dict(
        block_spawn_m=block,
        block_random_euler_rad=block_euler,
        target_spawn_m=target,
        archived_settled_xy_errors_m=errors,
        static_camera_count=len(config["static_camera_list"]),
        scope="Random draw reconstruction only. Settled state is not computed by this check.",
    )


def tick_zero_candidate():
    """Use the production wrapper and a mock physics backend, without seeding its tracker."""
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        backend, _, closed, world = fixture(root)
        backend.reset(dict(episode_id="tick-zero-probe", window=50, max_actions=800))
        backend.tick = 0
        world.tracker.last_stamp = None
        backend.guard.candidate = Mock(
            side_effect=G2Error(
                Status.COLLISION, "injected first candidate", geometry="candidate"
            )
        )
        backend.env.scene = SceneClock(
            backend.actual_scene, backend._stepped, backend._before_step
        )
        runtime = RuntimeReset(backend, {}, root, lambda *a: None, [])
        runtime.status = "matched"
        row = np.zeros((1, 14))
        row[:, [6, 13]] = 0.5
        try:
            runtime.execute(row, lambda _: None, lambda: 1)
        except G2Error as error:
            assert (
                error.status is Status.COLLISION
                and error.diagnostic["geometry"] == "candidate"
            )
        else:
            raise AssertionError("Expected injected candidate rejection")
        try:
            runtime.hold()
        except G2Error as error:
            assert error.status is Status.UNSYNCED
            proof_error = str(error)
        else:
            raise AssertionError("A missing current sample must prevent a stop receipt")
        finally:
            runtime.close(root)
        assert backend.tick == 0 and closed == [True] and not runtime.candidate_stop
        assert runtime.cleanup["hold_complete"] and runtime.cleanup["close_complete"]
        return dict(
            candidate_provenance_preserved=True,
            physics_steps=backend.tick,
            hold_and_close_called=True,
            stop_receipt_issued=False,
            stop_proof_error=proof_error,
            cleanup_failures=runtime.cleanup_failures,
            campaign_continuation_admissible=False,
            scope="Injected CPU test of current wrapper semantics; no claim of a physical collision at scene 100000.",
        )


def compare_reset(readback, archive):
    """Measure agreement to the archived GPU smoke for the same seed."""
    from g2_initial import matrix7
    from g2_kinematics import pose_error

    assert readback["cache"]["scene_seed"] == archive["scene_seed"]
    assert readback["cache"]["sha256"] == archive["cache_sha256"]
    initial = readback["initial"]
    actual = {
        **initial["objects_task"],
        **{
            f"{side}_robot_base": initial["robot_base_world"][side]
            for side in ("left", "right")
        },
    }
    errors = {}
    for name, saved in archive["initial_poses"].items():
        error = pose_error(
            matrix7(actual[name]), matrix7(saved["position"] + saved["quaternion_wxyz"])
        )
        errors[name] = dict(
            position_m=float(np.linalg.norm(error[:3])),
            angle_deg=float(np.degrees(np.linalg.norm(error[3:]))),
        )
    state_difference = float(
        np.max(
            np.abs(
                np.asarray(readback["model_state_from_native_getters"])
                - archive["drive_state"]
            )
        )
    )
    return dict(
        archived_vs_cpu_pose_errors=errors,
        drive_state_max_absolute_difference=state_difference,
        existing_pose_tolerance_m=0.001,
        existing_pose_tolerance_deg=0.1,
        pose_matches_within_existing_initial_tolerances=all(
            row["position_m"] <= 0.001 and row["angle_deg"] <= 0.1
            for row in errors.values()
        ),
        scope="Physical pose comparison against historical first-frame metadata only; no new rendering.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readback", type=Path)
    args = parser.parse_args()
    archive = archived_first_frame()
    comparison = (
        compare_reset(json.loads(args.readback.read_text()), archive)
        if args.readback
        else None
    )
    with args.output.open("x") as stream:
        json.dump(
            dict(
                archive=archive,
                spawn_rng=spawn_rng_check(archive),
                physical_cpu_comparison=comparison,
                tick_zero=tick_zero_candidate(),
                gpu_used=False,
                model_requests=0,
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
