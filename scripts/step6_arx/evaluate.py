"""Run the fixed Step 6 pair matrix in an already available simulator environment."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import signal
import sys
from time import monotonic, time_ns
import traceback

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/step6"


class ResourceCensored(Exception):
    pass


class InitialMismatch(Exception):
    pass


def install_safety(root):
    """Select the exact PR 5 runtime ahead of the older modules inherited from PR 6."""
    manifest = json.loads((DOCS / "pr5_runtime_sha256.json").read_text())
    for name, digest in manifest.items():
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"PR 5 runtime drift: {name}")
    sys.path.insert(0, str(root / "scripts"))


def interrupt(signum, _frame):
    raise ResourceCensored(f"received signal {signum}")


def compare_reset(a, b):
    import numpy as np
    from g2_initial import compare_initial

    errors = compare_initial(a["physics"], b["physics"],
                             position_m=1e-5, angle_deg=.002, joint_rad=1e-5, grip_fraction=1e-4)
    for key in a["readback"]:
        x, y = a["readback"][key], b["readback"][key]
        if key == "camera_config":
            if (x is None) != (y is None) or x is not None and (
                    x.keys() != y.keys() or any(
                        x[cam].keys() != y[cam].keys() or any(
                            not np.allclose(x[cam][k], y[cam][k], atol=1e-5, rtol=0) for k in x[cam])
                        for cam in x)):
                errors.append(key)
        elif not np.allclose(x, y, atol=1e-5, rtol=0):
            errors.append(key)
    return errors


def trial(args, plan, safety, seed, variant, out, uri, reference, model_reference):
    from g2_adapter import G2Error
    from g2_episode import json_value, save_json
    from robotwin_control_geometry import Status
    from openpi_client import msgpack_numpy

    from .metrics import Stages
    from .protocol import (actions_from, decode, record_wire, socket_for, validate_metadata)
    from .scene import CamerasUnavailable, open_scene

    start = monotonic()
    deadline = start + (600 if args.cpu_fixture or args.gpu_short else plan["timeout_s"])
    budget = 8 if args.cpu_fixture else plan["execute_cap"] if args.gpu_short else plan["max_actions"]

    def remaining():
        left = deadline - monotonic()
        if left <= 0:
            raise ResourceCensored("trial wall time limit")
        return left

    out.mkdir()
    wire = out / "wire"
    wire.mkdir()
    metrics = Stages()
    episode = f"seed-{seed:02d}-{variant}"
    record = dict(schema=plan["protocol"], episode=episode, scene_seed=seed, variant=variant,
                  source="protocol_fixture" if args.cpu_fixture else "step5_model",
                  rgb_status="GPU_UNVERIFIED" if args.headless else "NOT_READY",
                  policy_seed=plan["policy_seed"], execution_cap=plan["execute_cap"],
                  horizon=plan["horizon"], max_actions=budget, timeout_s=deadline - start,
                  prompt=plan["prompt"], query_rule="starts at zero each trial; increment per request",
                  safety_revision=plan["safety_revision"], checkpoint=None,
                  physics_ticks=0, policy_actions=0, policy_rows_attempted=0, queries=0,
                  task_success=None, native_success_at_stop=None, outcome="INFRASTRUCTURE_ERROR",
                  failure=None, cleanup_errors=[], initial=None, pair_match=None, video=None,
                  start_unix_ns=time_ns())
    if args.gpu_short:
        record["acceptance_only"] = True
    query = None
    phase = "scene_setup"
    scene = None
    last_recorded_tick = None
    with (out / "trajectory.jsonl").open("w", buffering=1) as trace:
        def emit(event):
            event = {"unix_ns": time_ns(), "query_index": query,
                     "action_index": record["policy_actions"], **event}
            trace.write(json.dumps(event, default=json_value, allow_nan=False) + "\n")
            return event

        def physics():
            nonlocal last_recorded_tick
            if last_recorded_tick != scene.tick:
                row = emit(dict(kind="physics", **scene.observe_physics(scene.state())))
                # Normalize arrays into the same representation as offline replay.
                metrics.step(json.loads(json.dumps(row, default=json_value, allow_nan=False)))
                last_recorded_tick = scene.tick

        def control_event(event):
            emit(dict(event, kind=event.get("kind", "controller_step")))
            if event.get("physics_step_count"):
                physics()

        resources = ExitStack()
        try:
            scene, setup = resources.enter_context(open_scene(args, plan, safety, seed, out, episode))
            record["setup"] = setup
            phase = "reset_admission"
            initial = scene.reset({"episode_id": episode})
            record["initial"] = {"physics": initial, "readback": scene.initial_extras()}
            record.update(physics_dt_s=scene.dt, control_period_s=scene.motion.period)
            save_json(out / "initial.json", record["initial"])
            physics()
            if variant == "lora":
                errors = compare_reset(reference, record["initial"]) if reference else ["frozen_initial_missing"]
                record["pair_match"] = not errors
                record["pair_mismatches"] = errors
                if errors:
                    raise InitialMismatch(", ".join(errors))
            if scene.success():
                raise InitialMismatch("native success predicate true at reset")

            phase = "policy_handshake"
            socket = resources.enter_context(socket_for(uri, min(20, remaining())))
            frame = socket.recv(timeout=min(30, remaining()))
            emit(dict(kind="policy_handshake", payload=record_wire(wire, 0, "metadata", frame)))
            metadata = decode(frame)
            validate_metadata(metadata, variant, plan, args.cpu_fixture)
            record["checkpoint"] = metadata
            if model_reference is not None:
                old = model_reference["step6_identity"]
                new = metadata["step6_identity"]
                if any(old[k] != new[k] for k in old if k != "selected_checkpoint_sha256"):
                    raise InitialMismatch("policy normalization or frozen model provenance differs")
            elif variant == "lora" and not args.cpu_fixture:
                raise InitialMismatch("missing frozen model provenance")
            while record["policy_actions"] < budget and not scene.success():
                phase = "observation"
                query = record["queries"]
                request = scene.request(query)
                stamp = asdict(scene.last_capture.stamp)
                if not args.headless:
                    record["rgb_status"] = "OBSERVED"
                packed = msgpack_numpy.packb(request)
                emit(dict(kind="policy_request", stamp=stamp, sampled_unix_ns=scene.sampled_ns,
                          rgb_source=record["rgb_status"], command=request["state"],
                          camera_config=scene.camera_config,
                          payload=record_wire(wire, query, "request", packed)))
                phase = "policy_response"
                socket.send(packed)
                record["queries"] += 1
                frame = socket.recv(timeout=min(remaining(), 380 if args.gpu_short else 180))
                emit(dict(kind="policy_response", stamp=stamp, payload=record_wire(wire, query, "response", frame)))
                chunk = actions_from(decode(frame))
                if scene.tick != stamp["tick"]:
                    raise G2Error(Status.UNSYNCED, "policy response addressed a stale observation")
                phase = "safe_execution"
                n = min(plan["execute_cap"], budget - record["policy_actions"])
                for row_i in range(n):
                    remaining()
                    record["policy_rows_attempted"] += 1
                    before_tick = scene.tick
                    event = emit(dict(kind="row_attempt", row_in_horizon=row_i, tick=before_tick,
                                      sim_s=before_tick * scene.dt, target=chunk[row_i]))
                    metrics.record("collision_rejection", False, event)
                    reason = scene.execute_native_row(chunk[row_i], remaining, control_event)
                    record["policy_actions"] += 1
                    emit(dict(kind="row_complete", row_in_horizon=row_i, first_tick=before_tick,
                              tick=scene.tick, sim_s=scene.tick * scene.dt, replan=reason))
                    if reason or scene.success():
                        break
                emit(dict(kind="window_end", consumed_rows=row_i + 1,
                          discarded_rows=plan["horizon"] - row_i - 1, tick=scene.tick))
                if args.gpu_short:
                    break
            record["task_success"] = scene.success()
            record["outcome"] = "TASK_SUCCESS" if record["task_success"] else "ACTION_LIMIT"
        except (Exception, KeyboardInterrupt) as error:
            status = error.status.value if isinstance(error, G2Error) else type(error).__name__
            diagnostic = error.diagnostic if isinstance(error, G2Error) else {}
            failure = dict(phase=phase, status=status, detail=str(error), diagnostic=diagnostic)
            record["failure"] = failure
            if phase == "safe_execution" and isinstance(error, G2Error):
                if error.status == Status.TIMEOUT:
                    record["outcome"] = "RESOURCE_CENSORED"
                elif error.status in (Status.COLLISION, Status.LIMIT, Status.PLAN_FAILED,
                                       Status.VELOCITY, Status.IK_FAILED):
                    record.update(outcome="SAFETY_STOP", task_success=False)
                elif error.status == Status.BACKEND_ERROR and (
                        diagnostic.get("geometry") == "actual" or "contact_state" in diagnostic):
                    record.update(outcome="SAFETY_STOP", task_success=False)
            if isinstance(error, (ResourceCensored, KeyboardInterrupt)) or (
                    isinstance(error, TimeoutError) and monotonic() >= deadline):
                record["outcome"] = "RESOURCE_CENSORED"
            if isinstance(error, InitialMismatch):
                record["outcome"] = "INITIAL_OR_PAIR_MISMATCH"
            if isinstance(error, CamerasUnavailable):
                record["rgb_status"] = "GPU_UNVERIFIED" if args.cpu_fixture else "NOT_READY"
            if record["outcome"] == "INFRASTRUCTURE_ERROR":
                failure["traceback"] = traceback.format_exc()
            event = emit(dict(kind="stop", tick=scene.tick if scene else None,
                              sim_s=scene.tick * scene.dt if scene else None, **failure))
            metrics.safety(event)
        finally:
            if scene is not None and scene.episode is not None:
                record["physics_ticks"] = scene.tick
                # Even exceptions between scene.step() and callback preserve physical readback.
                try:
                    if record["initial"] is not None:
                        physics()
                    record["native_success_at_stop"] = scene.success()
                except Exception as error:
                    record["cleanup_errors"].append(f"terminal readback: {error}")
                if record["task_success"] is not True:
                    try:
                        scene.hold()
                        emit(dict(kind="terminal_hold", tick=scene.tick, command=scene.command(),
                                  finger_drive_targets_m=scene._drive_fingers()))
                    except Exception as error:
                        record["cleanup_errors"].append(f"hold: {error}")
            try:
                resources.close()
            except Exception as error:
                record["cleanup_errors"].append(f"close: {error}")
            if (out / "head.mp4").is_file() and not args.headless:
                record["video"] = "head.mp4"
            record["analysis"] = metrics.summary()
            record["wall_s"] = monotonic() - start
            if record["cleanup_errors"]:
                record["task_success"] = None
                record["outcome"] = "INFRASTRUCTURE_ERROR"
            save_json(out / "receipt.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pr5-root", "pace-root", "asset-audit", "output"):
        parser.add_argument(f"--{name}", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--seed", type=int, action="append", help="subset of frozen matrix; default seeds 0–4")
    parser.add_argument("--variant", choices=("both", "frozen", "lora"), default="both",
                        help="use two passes if only one model service fits")
    parser.add_argument("--frozen-url")
    parser.add_argument("--lora-url")
    parser.add_argument("--gpu-short", action="store_true",
                        help="one real H50 query per variant, followed by up to K4 guarded actions")
    parser.add_argument("--cpu-fixture", action="store_true")
    parser.add_argument("--headless", action="store_true", help="CPU fixture only; keeps native PhysX without RGB")
    parser.add_argument("--invalid-variant", choices=("frozen", "lora"), help="CPU fixture: reject at query 1")
    parser.add_argument("--dataset", type=lambda p: Path(p).resolve(), help="CPU fixture: use real Step 5 norm stats")
    args = parser.parse_args()
    if args.gpu_short and (args.cpu_fixture or args.seed != [0] or args.variant == "both"):
        parser.error("GPU short acceptance requires --seed 0 and one variant per service process")
    if args.cpu_fixture:
        if args.frozen_url or args.lora_url or args.dataset is None or args.variant != "both":
            parser.error("CPU fixture needs --dataset; it owns both loopback endpoints")
    elif (args.headless or args.invalid_variant or args.dataset
          or args.variant != "lora" and not args.frozen_url
          or args.variant != "frozen" and not args.lora_url):
        parser.error("real evaluation needs a URL for each selected variant; fixture options cannot be used")
    install_safety(args.pr5_root)
    if sha256(args.asset_audit.read_bytes()).hexdigest() != "5ebfc07c82814757168bdb3f2b7ad4a9bec18f58afc28318f49e61af117122f2":
        parser.error("ARX asset audit must match the pinned PR 5 manifest")
    from pace_runtime import PACE_REVISION, require_checkout

    require_checkout(args.pace_root, PACE_REVISION)
    plan = json.loads((DOCS / "pairs.json").read_text())
    safety = json.loads((DOCS / "arx_safety_profile.json").read_text())
    seeds = plan["scene_seeds"] if args.seed is None else args.seed
    if not set(seeds) <= set(plan["scene_seeds"]) or len(seeds) != len(set(seeds)):
        parser.error("select distinct seeds from 0–4")
    campaign = dict(plan=plan, seeds=seeds, fixture=args.cpu_fixture)
    if args.gpu_short:
        campaign["acceptance_only"] = True
    if args.variant == "lora":
        if json.loads((args.output / "campaign.json").read_text()) != campaign:
            parser.error("LoRA pass must use the same frozen campaign directory and seeds")
    else:
        args.output.mkdir(parents=True)
        (args.output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with ExitStack() as stack:
        urls = dict(frozen=args.frozen_url, lora=args.lora_url)
        if args.cpu_fixture:
            from .fixture import fixture_adapter, serve_fixture
            print("Step 6 verifying the existing dataset and loading normalization", flush=True)
            adapter = fixture_adapter(args.dataset, plan)
            urls = {v: stack.enter_context(serve_fixture(args, v, plan, adapter)) for v in plan["variants"]}
        def run(seed, variant, initial=None, checkpoint=None):
            print(f"Step 6 reset seed={seed} variant={variant}", flush=True)
            receipt = trial(args, plan, safety, seed, variant, args.output / f"seed-{seed:02d}-{variant}",
                            urls[variant], initial, checkpoint)
            print(json.dumps({k: receipt[k] for k in ("episode", "outcome", "physics_ticks", "queries")}), flush=True)
            return receipt

        def stop(receipt):
            return bool(receipt["cleanup_errors"]) or (receipt["outcome"] == "RESOURCE_CENSORED"
                and receipt["failure"]["detail"].startswith("received signal"))

        for seed in seeds:
            if args.variant == "lora":
                frozen_path = args.output / f"seed-{seed:02d}-frozen" / "receipt.json"
                if not frozen_path.is_file():
                    continue
                frozen = json.loads(frozen_path.read_text())
            else:
                frozen = run(seed, "frozen")
            if stop(frozen):
                break
            if args.variant != "frozen" and stop(run(seed, "lora", frozen["initial"], frozen["checkpoint"])):
                break


if __name__ == "__main__":
    main()
