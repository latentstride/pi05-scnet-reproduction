"""Independent author_native_h50 diagnostic. No campaign or scheduler entry."""

from contextlib import contextmanager
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
import re
import random
from time import monotonic

import numpy as np

from g2_adapter import G2Error, Snapshot, Stamp, read_snapshot
from g2_assets import load_arms
from g2_candidate_stop import (
    candidate_failure,
    failure_sha256,
    measure_stop,
    verified_candidate_stop,
)
from g2_capture import capture_arms
from g2_episode import json_value
from g2_final_campaign import verified_scene_cleanup
from g2_initial import capture_initial, compare_initial, digest_json
from g2_native_backend import NativeBackend, SceneClock
from g2_task_runtime import pace_context
from g2_unified_exchange import hash_array
from pace_runtime import (
    PACE_REVISION,
    TASK,
    require_checkout,
    validate_request,
    validate_response,
)
from robotwin_control_geometry import Status, pose_from_wxyz

MODE = "author_native_h50"
SEED = 100000
WINDOW = 50
MAX_ACTIONS = 800
ARCHIVE_CONFIG = "04f2e041cedb07a2eae69425175c60d235ed086095cce10600dfb9e492a8bae6"
ASSET_AUDIT = "5ebfc07c82814757168bdb3f2b7ad4a9bec18f58afc28318f49e61af117122f2"
PREFLIGHT = "b1ab4f6201174a872d00468489968313bb445f24be7ab8c9b7ebfe73849a3787"


class ProtocolDeviation(Exception):
    pass


def locked_json(path, digest):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"frozen input differs: {path.name}")
    return json.loads(data)


def contact_readback(scene):
    """Return engine manifolds, including an empty list only on a successful read."""
    contacts = scene.get_contacts()
    if not isinstance(contacts, (list, tuple)):
        raise G2Error(
            Status.UNSYNCED, "PhysX contact measurement unavailable", geometry="actual"
        )
    rows = []
    for contact in contacts:
        if len(contact.bodies) != 2:
            raise G2Error(
                Status.UNSYNCED,
                "PhysX contact identities incomplete",
                geometry="actual",
            )
        rows.append(
            dict(
                bodies=[
                    (int(body.entity.per_scene_id), body.entity.name)
                    for body in contact.bodies
                ],
                points=[
                    dict(
                        position=p.position.tolist(),
                        normal=p.normal.tolist(),
                        impulse=p.impulse.tolist(),
                        separation=float(p.separation),
                    )
                    for p in contact.points
                ],
            )
        )
    # PhysX does not contract an ordering of manifolds or their points.
    for row in rows:
        row["points"].sort(key=digest_json)
    rows.sort(key=digest_json)
    return contacts, rows


class AuthorBackend(NativeBackend):
    """Use the existing step gates without the project prefix/control wrapper."""

    def __init__(self, env, arms, initial, guard, episode, log, remaining):
        super().__init__(env, arms, np.eye(4), initial, None, guard=guard)
        self.episode, self.log, self.remaining = episode, log, remaining
        self.completed = 0
        self.query_starts = []
        self.initial_measurement = None
        self.grip_max_clip = None

    @property
    def take_action_cnt(self):
        return self.env.take_action_cnt

    @property
    def step_lim(self):
        return self.env.step_lim

    @property
    def eval_success(self):
        return self.env.eval_success

    def get_instruction(self):
        return self.env.get_instruction()

    def get_obs(self):
        if self.completed != self.take_action_cnt or self.completed % WINDOW:
            raise ProtocolDeviation("observation attempted inside a Fixed H50 block")
        self.remaining()
        before = self.tick
        obs = self.env.get_obs()
        if before != self.tick:
            raise ProtocolDeviation("observation advanced the simulator")
        self.query_starts.append(self.completed)
        return obs

    def sample_initial(self, reset_steps):
        if (
            self.tick != 0
            or reset_steps <= 0
            or self.guard.collision.tracker.last_stamp is not None
        ):
            raise G2Error(
                Status.UNSYNCED,
                "initial PhysX contact clock is unverified",
                geometry="actual",
            )
        stamp = Stamp(self.episode, 0, 0.0)
        snapshot = Snapshot(
            stamp, capture_arms(self.env.robot, stamp), {}, self.world_task
        )
        q, fingers, _, tools = read_snapshot(snapshot, self.arms)
        block = self.env.box.actor.get_pose()
        contacts, rows = contact_readback(self.actual_scene)
        self.guard.measured(
            stamp, q, fingers, tools, pose_from_wxyz(block.p, block.q), contacts
        )
        self.guard.previous_drive = self._drive()[1].copy()
        measured = measure_stop(self)
        _, after = contact_readback(self.actual_scene)
        if self.tick != 0 or digest_json(rows) != digest_json(after):
            raise G2Error(
                Status.UNSYNCED, "initial PhysX measurement changed", geometry="actual"
            )
        self.initial_measurement = dict(
            source="sapien.Scene.get_contacts after last real setup_demo physics step",
            status="measured",
            stamp=asdict(stamp),
            author_stability_physics_steps=reset_steps,
            state_sha256=measured["state_sha256"],
            contact_sha256=digest_json(rows),
            manifold_count=len(rows),
            contacts=rows,
        )
        self.log(dict(kind="initial_physical_measurement", **self.initial_measurement))

    def take_action(self, action):
        self.remaining()
        before_count, before_tick = self.take_action_cnt, self.tick
        if before_count != self.completed or self.step_lim != MAX_ACTIONS:
            raise ProtocolDeviation("author action counter or limit changed")
        self.action_index = self.completed
        row = action.copy()
        self.log(
            dict(
                kind="author_action",
                index=self.completed,
                tick=self.tick,
                row=row,
                row_sha256=hash_array(row),
            )
        )
        # The author's set_gripper performs its own [0,1] saturation. It is
        # applied inside its controller, not to the raw model row here.
        grips = row[[6, 13]]
        if np.any(np.abs(grips - np.clip(grips, 0, 1)) > self.grip_max_clip):
            raise G2Error(
                Status.LIMIT,
                "raw author gripper exceeds existing clip allowance",
                geometry="candidate",
            )
        for arm, values in zip(self.arms, row.reshape(2, 7)):
            if np.any(values[:6] < arm.chain.limits[:, 0]) or np.any(
                values[:6] > arm.chain.limits[:, 1]
            ):
                raise G2Error(
                    Status.LIMIT,
                    "author joint endpoint outside existing limits",
                    geometry="candidate",
                )
        self.env.take_action(row, action_type="qpos")
        if (
            self.take_action_cnt != before_count + 1
            or not np.array_equal(row, action)
            or self.tick <= before_tick
        ):
            raise ProtocolDeviation(
                "native action consumption, row or physical clock diverged"
            )
        self.completed += 1
        self.log(
            dict(
                kind="author_action_complete",
                index=before_count,
                tick=self.tick,
                physics_steps=self.tick - before_tick,
            )
        )

    def terminal_sample(self, *, held=False):
        _, contacts = contact_readback(self.actual_scene)
        proof = measure_stop(self, held=held)
        _, after = contact_readback(self.actual_scene)
        contact_hash = digest_json(contacts)
        if contact_hash != digest_json(after):
            raise G2Error(
                Status.UNSYNCED, "terminal PhysX contacts changed", geometry="actual"
            )
        if self.tick == 0 and (
            self.initial_measurement is None
            or (proof["state_sha256"], contact_hash)
            != (
                self.initial_measurement["state_sha256"],
                self.initial_measurement["contact_sha256"],
            )
        ):
            raise G2Error(
                Status.UNSYNCED,
                "terminal state is not the sampled reset",
                geometry="actual",
            )
        return dict(
            proof,
            contact_sha256=contact_hash,
            contact_manifold_count=len(contacts),
            contact_source="sapien.Scene.get_contacts",
            contact_sample_measured=True,
        )


def native_guard(env, arms, samples, config):
    from mplib.sapien_utils.conversion import convert_object_name
    from g2_collision import JointCollisionWorld
    from g2_contact import ContactLimits, ContactWorld
    from g2e_native_guard import NativeGuard, NativeReleaseTracker

    entities = [env.robot.left_entity, env.robot.right_entity]
    if entities[0] is not entities[1]:
        raise ValueError("expected the shared Aloha articulation")
    collision = JointCollisionWorld.from_scene(
        env.scene, entities, [a.chain.names for a in arms], [a.grip.names for a in arms]
    )
    block = env.box.actor
    tracker = NativeReleaseTracker.from_scene(env.robot, block, arms, ContactLimits())
    actors = env.scene.get_all_actors()
    world = ContactWorld(
        collision,
        tracker,
        arms,
        [sample.world_base for sample in samples],
        convert_object_name(block),
        [convert_object_name(a) for a in actors],
        [
            [entity.find_joint_by_name(n).child_link.name for n in arm.grip.names]
            for entity, arm in zip(entities, arms)
        ],
        [convert_object_name(a) for a in actors if a.name in ("table", "target_box")],
        world_task=np.eye(4),
    )
    return NativeGuard(
        world,
        arms,
        config["collision_joint_step_rad"],
        config["tracking_position_limit_m"],
        config["tracking_angle_limit_deg"],
    )


@contextmanager
def author_scene(
    pace, release, reference, episode, log, remaining, *, configure_cpu=None
):
    from sapien.physx import PhysxRigidDynamicComponent

    require_checkout(pace, PACE_REVISION)
    baseline = locked_json(reference, PREFLIGHT)
    config = locked_json(
        release / "configs/unified-seed0-native-w50.json", ARCHIVE_CONFIG
    )
    audit = release / "asset-audit.json"
    locked_json(audit, ASSET_AUDIT)
    arms = load_arms(
        pace / "assets", audit, "aloha-agilex", config["calibration"]["aloha-agilex"]
    )
    with pace_context(pace):
        # The CPU acceptance substitutes rendering and unused CuRobo warm-up
        # only. The production caller never provides this adapter.
        from contextlib import nullcontext

        with configure_cpu(pace, audit) if configure_cpu else nullcontext():
            from pace.robotwin import (
                load_cached_scenes,
                load_environment,
                restore_cached_scene_state,
                sample_instruction,
            )
            from pace.rollout import run_episode

            cached, info = load_cached_scenes(TASK, 0, 1)
            if (
                cached != [{"scene_seed": SEED, "episode_info": {}}]
                or info["sha256"] != baseline["cache"]["sha256"]
            ):
                raise ValueError("author scene cache mismatch")
            env, settings = load_environment(TASK)
            if settings["embodiment"] != ["aloha-agilex"]:
                raise ValueError("author embodiment mismatch")
            if configure_cpu:
                configure_cpu.settings(settings)
            check_stable = env.check_stable
            reset_steps = 0

            def count_reset_step():
                nonlocal reset_steps
                reset_steps += 1

            def check_stable_with_clock():
                scene = env.scene
                env.scene = SceneClock(scene, count_reset_step)
                try:
                    return check_stable()
                finally:
                    env.scene = scene

            try:
                env.check_stable = check_stable_with_clock
                random.seed(SEED)
                env.setup_demo(now_ep_num=0, seed=SEED, is_test=True, **settings)
                if reset_steps != 2500:
                    raise ValueError("author reset physics witness incomplete")
                restore_cached_scene_state(TASK, env, cached[0]["episode_info"])
                env.set_instruction(
                    sample_instruction(
                        TASK, {"info": cached[0]["episode_info"]}, 100, "unseen"
                    )
                )
                if (
                    env.step_lim != MAX_ACTIONS
                    or env.take_action_cnt
                    or env.eval_success
                    or env.check_success()
                ):
                    raise ProtocolDeviation(
                        "unexpected author reset, action limit or predicate"
                    )
                mass = float(
                    env.box.actor.find_component_by_type(
                        PhysxRigidDynamicComponent
                    ).mass
                )
                if abs(mass - baseline["block_mass_kg"]) > 1e-8:
                    raise ValueError("author block mass differs")
                samples = capture_arms(env.robot, Stamp(episode, 0, 0.0))
                bound_arms = tuple(
                    replace(a, instance=s.instance) for a, s in zip(arms, samples)
                )
                initial = capture_initial(
                    env,
                    bound_arms,
                    np.eye(4),
                    episode,
                    digest_json(cached[0]),
                    "author-native-h50-reset",
                    False,
                )
                mismatch = compare_initial(initial, baseline["initial"])
                if mismatch:
                    raise ValueError(f"author reset mismatch: {mismatch}")
                guard = native_guard(env, bound_arms, samples, config)
                backend = AuthorBackend(
                    env, bound_arms, initial, guard, episode, log, remaining
                )
                backend.grip_max_clip = config["native_grip_max_clip"]
                backend.sample_initial(reset_steps)
                log(
                    dict(
                        kind="author_reset_admitted",
                        scene_seed=SEED,
                        cache_sha256=info["sha256"],
                        initial=initial,
                        block_mass_kg=mass,
                        prompt=env.get_instruction(),
                    )
                )
                yield backend, run_episode
            finally:
                if hasattr(env, "scene"):
                    env.close_env(clear_cache=configure_cpu is None)


class RecordedPolicy:
    """Save the actual first request and each original chunk at the boundary."""

    def __init__(self, policy, output, log, backend, *, require_real_rgb):
        self.policy, self.output, self.log, self.backend = policy, output, log, backend
        self.require_real_rgb = require_real_rgb
        self.queries = 0

    def infer(self, request):
        validate_request(request)
        if (
            len(self.backend.query_starts) != self.queries + 1
            or request["prompt"] != self.backend.get_instruction()
        ):
            raise ProtocolDeviation("unbound author request or prompt")
        if any(p.shape != (3, 240, 320) for p in request["images"].values()):
            raise ValueError("unexpected author camera dimensions")
        if self.require_real_rgb and any(
            p.min() == p.max() for p in request["images"].values()
        ):
            raise ValueError("author camera frame is constant")
        q = self.queries
        with (self.output / f"request-{q:02d}.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                state=request["state"],
                prompt=request["prompt"],
                **request["images"],
            )
        self.log(
            dict(
                kind="author_query",
                query=q,
                index=self.backend.completed,
                tick=self.backend.tick,
            )
        )
        self.queries += 1
        response = self.policy.infer(request)
        actions = validate_response(response)
        with (self.output / f"chunk-{q:02d}.npy").open("xb") as stream:
            np.save(stream, actions, allow_pickle=False)
        self.log(
            dict(
                kind="author_chunk",
                query=q,
                sha256=hash_array(actions),
                rng=response.get("author_rng"),
            )
        )
        return response


def run_diagnostic(
    pace,
    release,
    reference,
    output,
    episode,
    policy_factory,
    *,
    deadline,
    configure_cpu=None,
    inject=None,
):
    """Shared runtime for the live entry and the CPU physics acceptance."""
    if not episode or not np.isfinite(deadline):
        raise ValueError("diagnostic requires an episode and finite wall deadline")
    output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    deadline = min(deadline, started + 1150.0)
    cleanup = dict(
        scene_opened=False,
        hold_complete=False,
        close_complete=False,
        scene_context_closed=False,
    )
    receipt = dict(
        mode=MODE,
        episode_id=episode,
        cleanup=cleanup,
        cleanup_failures=[],
        fresh_reset_verified=False,
        success=False,
        category="infrastructure_failure",
        policy_kind="cpu_substitute" if configure_cpu else "checkpoint",
    )
    backend, proof = None, None

    def remaining():
        seconds = deadline - monotonic()
        if seconds <= 0:
            raise G2Error(Status.TIMEOUT, "author trial wall-clock budget reached")
        return seconds

    with (output / "events.jsonl").open("x", buffering=1) as stream:

        def log(event):
            stream.write(json.dumps(event, default=json_value, allow_nan=False) + "\n")

        try:
            remaining()
            with author_scene(
                pace,
                release,
                reference,
                episode,
                log,
                remaining,
                configure_cpu=configure_cpu,
            ) as (backend, rollout):
                cleanup["scene_opened"] = receipt["fresh_reset_verified"] = True
                receipt["prompt"] = backend.get_instruction()
                receipt["initial_measurement"] = {
                    k: v
                    for k, v in backend.initial_measurement.items()
                    if k != "contacts"
                }
                try:
                    if inject:
                        inject(backend)
                    with policy_factory() as policy:
                        if hasattr(policy, "reset_receipt"):
                            log(
                                dict(
                                    kind="author_model_rng_reset",
                                    receipt=policy.reset_receipt,
                                )
                            )
                        recorded = RecordedPolicy(
                            policy,
                            output,
                            log,
                            backend,
                            require_real_rgb=configure_cpu is None,
                        )
                        result = rollout(backend, recorded, lambda _: WINDOW)
                        if (
                            result["steps"] != backend.completed
                            or result["queries"] != recorded.queries
                        ):
                            raise ProtocolDeviation("author rollout counters diverged")
                        if result["success"] != backend.success() or not (
                            result["success"] or result["steps"] == MAX_ACTIONS
                        ):
                            raise ProtocolDeviation(
                                "author rollout stopped without native success or action limit"
                            )
                        receipt.update(
                            category="diagnostic_complete",
                            outcome="SUCCESS"
                            if result["success"]
                            else "TASK_INCOMPLETE",
                            success=result["success"],
                        )
                except G2Error as error:
                    failure = dict(
                        code=error.status.value,
                        detail=str(error),
                        diagnostic=error.diagnostic,
                    )
                    receipt["result"] = dict(failure=failure)
                    if (
                        error.status == Status.TIMEOUT
                        and error.diagnostic.get("geometry") != "actual"
                    ):
                        receipt["outcome"] = "RESOURCE_CENSORED"
                    else:
                        receipt.update(
                            category="actual_safety_stop", outcome="SAFETY_STOP"
                        )
                    if candidate_failure(receipt["result"]):
                        try:
                            proof = backend.terminal_sample()
                        except Exception as exc:
                            receipt["cleanup_failures"].append(
                                f"terminal sample: {exc}"
                            )
                except ProtocolDeviation as error:
                    receipt.update(outcome="PROTOCOL_DEVIATION", detail=str(error))
                finally:
                    tick = backend.tick
                    try:
                        backend.hold()
                        if backend.tick != tick:
                            raise RuntimeError("hold advanced author physics")
                        cleanup["hold_complete"] = True
                        if proof is not None:
                            after = backend.terminal_sample(held=True)
                            keys = ("state_sha256", "contact_sha256", "tick")
                            if (
                                any(proof[k] != after[k] for k in keys)
                                or proof["expected_hold_finger_m"]
                                != after["finger_drive_m"]
                            ):
                                raise RuntimeError(
                                    "candidate hold changed the paused state or expected drive"
                                )
                            proof = dict(
                                after,
                                hold_complete=True,
                                failure_sha256=failure_sha256(receipt["result"]),
                            )
                    except Exception as exc:
                        proof = None
                        receipt["cleanup_failures"].append(f"hold: {exc}")
                    receipt.update(
                        completed_actions=backend.completed,
                        native_action_count=backend.take_action_cnt,
                        physics_steps=backend.tick,
                        query_action_counts=backend.query_starts,
                    )
            if backend.tick != tick:
                raise RuntimeError("author scene close advanced physics")
            cleanup["close_complete"] = cleanup["scene_context_closed"] = True
            if proof and not receipt["cleanup_failures"]:
                receipt["candidate_stop"] = dict(
                    proof, close_complete=True, scene_context_closed=True
                )
                if verified_candidate_stop(
                    receipt["candidate_stop"], episode, receipt["result"]
                ):
                    receipt["category"] = "controlled_candidate_stop"
        except Exception as error:
            if isinstance(error, ProtocolDeviation):
                receipt["outcome"] = "PROTOCOL_DEVIATION"
            receipt["detail"] = f"{type(error).__name__}: {error}"
            receipt["category"] = "infrastructure_failure"
        if receipt["cleanup_failures"]:
            receipt["category"] = "infrastructure_failure"
        receipt["elapsed_s"] = monotonic() - started
        log(dict(kind="author_terminal", **receipt))
    (output / "receipt.json").write_text(
        json.dumps(receipt, default=json_value, indent=2, allow_nan=False) + "\n"
    )
    return receipt


def may_continue(
    receipt, expected_episode, *, process_group_reaped, shared_service_alive
):
    """Continuation gate for a future host. Next trial still needs fresh admission."""
    if (
        process_group_reaped is not True
        or shared_service_alive is not True
        or receipt.get("mode") != MODE
        or receipt.get("episode_id") != expected_episode
        or not verified_scene_cleanup(receipt)
    ):
        return False
    if receipt["category"] == "diagnostic_complete":
        return receipt.get("outcome") in ("SUCCESS", "TASK_INCOMPLETE")
    proof = receipt.get("candidate_stop")
    return (
        receipt["category"] == "controlled_candidate_stop"
        and candidate_failure(receipt.get("result") or {})
        and verified_candidate_stop(proof, expected_episode, receipt["result"])
        and proof.get("contact_sample_measured") is True
        and proof.get("contact_source") == "sapien.Scene.get_contacts"
        and type(proof.get("contact_manifold_count")) is int
        and proof["contact_manifold_count"] >= 0
        and isinstance(proof.get("contact_sha256"), str)
        and re.fullmatch("[a-f0-9]{64}", proof["contact_sha256"]) is not None
        and (
            proof["tick"] != 0
            or (
                receipt.get("initial_measurement", {}).get("status") == "measured"
                and all(
                    proof[key] == receipt["initial_measurement"].get(key)
                    for key in ("state_sha256", "contact_sha256")
                )
            )
        )
    )


def author_native_h50(
    pace, release, reference, output, episode, uri, startup, *, deadline
):
    """Run inside a future, explicitly admitted campaign's existing allocation.

    This callable has no scheduler, roster, authorization or automatic retry
    entry. The host must admit this diagnostic before calling it.
    """
    from g2_author_native_h50_service import shared_connection
    from g2_locked_task import verify_model_receipt

    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise PermissionError(
            "author_native_h50 requires an existing approved allocation"
        )
    plan = json.loads((release / "unified-plan.json").read_text())
    manifest = release / "pace_download_manifest.json"
    if (
        sha256(manifest.read_bytes()).hexdigest()
        != "835d60676236e69276032f7bc1ea4837eb6fcda6d4f3dbab2043965063a3d97b"
    ):
        raise ValueError("author H50 requires the frozen checkpoint manifest")
    verify_model_receipt(plan["checkpoint"], manifest, startup)
    end = min(deadline, monotonic() + 1150.0)

    def remaining():
        seconds = end - monotonic()
        if seconds <= 0:
            raise G2Error(Status.TIMEOUT, "author shared-service deadline reached")
        return seconds

    return run_diagnostic(
        pace,
        release,
        reference,
        output,
        episode,
        shared_connection(uri, episode, remaining),
        deadline=end,
    )
