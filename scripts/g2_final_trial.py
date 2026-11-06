"""Run one locked admission or diagnostic inside an authorized allocation."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

from g2_adapter import G2Error, images_of
from g2_episode import json_value
from g2_final_deploy import authorization
from g2_final_expert import execute_expert, expert_prefix
from g2_locked_task import (
    InitialMismatch,
    VerifiedReset,
    verify_model_receipt,
    verify_real_scene,
)
from g2_remaining_conditions import run_condition
from g2_task_runtime import inspect_assets, real_scene, verify_runtime
from g2e_host_supervisor import interrupt_signals
from g2e_model_entry import reset_remote
from pace_g1_orchestrate import persist
from pace_runtime import open_policy, validate_request, validate_response
from robotwin_control_geometry import Status


def read(path):
    return json.loads(path.read_text())


def classify(result):
    failure = result.get("failure") or {}
    detail = failure.get("detail", "").lower()
    diagnostic = failure.get("diagnostic") or {}
    if (
        diagnostic.get("geometry") == "actual"
        or any(
            s in detail
            for s in (
                "measured state",
                "actual reset state",
                "measured native",
                "object slipped",
                "lost bilateral",
                "measured tcp",
                "unsynced",
            )
        )
        or failure.get("code") == "unsynced"
    ):
        return "actual_safety_stop"
    if result["outcome"] in (
        "timeout",
        "interrupted",
        "policy_failed",
        "environment_error",
        "export_failed",
    ):
        return "infrastructure_failure"
    if diagnostic.get("geometry") == "candidate":
        return "diagnostic_complete"
    if result["outcome"] in ("control_error", "joint_limit") and not any(
        s in detail
        for s in (
            "candidate",
            "predicted finger start",
            "donor opening",
            "scene changed after planning",
        )
    ):
        return "actual_safety_stop"
    return "diagnostic_complete"


def run(args):
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or os.environ.get("SLURM_JOB_NAME") != "cpf-pace-g2":
        raise PermissionError("an existing G2 allocation is required")
    release = args.release.resolve()
    approval = Path(os.environ["G2_FINAL_AUTHORIZATION"])
    authorization(release, approval, job, starting=False)
    if approval.with_suffix(".used").read_text().strip() != job:
        raise PermissionError("allocation authorization is not active")
    plan = read(release / "unified-plan.json")
    if job == plan["protected_existing_job"]:
        raise PermissionError("the historical allocation is protected")
    row = next(r for r in plan["trials"] if r["episode_id"] == args.episode)
    binding = plan["trial_bindings"][args.episode]
    config = read(release / "configs" / f"{args.episode}.json")
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = args.output / "episode"
    source = read(release / "source-audit.json")
    assets = args.pace / "assets"
    cleanup_failures = []

    class RuntimeReset(VerifiedReset):
        def hold(self):
            try:
                return self.backend.hold()
            except Exception as error:
                cleanup_failures.append(f"hold: {error}")
                raise

        def close(self, output):
            try:
                return self.backend.close(output)
            except Exception as error:
                cleanup_failures.append(f"close: {error}")
                raise

    @contextmanager
    def factory(spec):
        with real_scene(
            config,
            spec,
            args.pace,
            assets,
            release / "asset-audit.json",
            source,
            evidence,
        ) as (backend, adapter):
            yield (
                RuntimeReset(
                    backend, binding["expected_initial"], evidence, verify_real_scene
                ),
                adapter,
            )

    receipt = dict(episode_id=args.episode, category="infrastructure_failure")
    started = time.monotonic()
    try:
        with interrupt_signals():
            verify_runtime(args.pace, source)
            verify_model_receipt(
                plan["checkpoint"],
                release / "pace_download_manifest.json",
                args.model_startup,
            )
            try:
                inspect_assets(
                    config,
                    assets,
                    release / "asset-audit.json",
                    native_only=row["embodiment"] == "aloha-agilex",
                )
            except (ValueError, OSError) as error:
                if not args.probe:
                    raise
                receipt.update(category="not_admitted", detail=str(error))
                return
            if args.probe:
                evidence.mkdir()
                with factory(config) as (backend, adapter):
                    try:
                        backend.reset(config)
                        receipt["physical_admitted"] = True
                        snap, state = backend.capture()
                        if adapter is None:
                            request = dict(
                                state=state,
                                images=images_of(snap),
                                prompt=config["prompt"],
                            )
                        else:
                            result = adapter.observe(snap, config["prompt"], 30)
                            if not result.ok:
                                receipt.update(
                                    category="not_admitted", detail=result.detail
                                )
                                return
                            request = result.payload
                        validate_request(request)
                        with open_policy(args.uri, "checkpoint", 390) as policy:
                            persist(
                                args.output / "rng-reset.json",
                                reset_remote(
                                    policy, config["policy_rng_seed"], args.episode
                                ),
                            )
                            validate_response(policy.infer(request))
                        receipt.update(category="admitted")
                    finally:
                        backend.close(evidence)
            elif "expert" in row["method"]:
                evidence.mkdir()
                end = time.monotonic() + config["timeout_s"]

                def remaining():
                    seconds = end - time.monotonic()
                    if seconds <= 0:
                        raise G2Error(Status.TIMEOUT, "expert deadline")
                    return seconds

                with (
                    factory(config) as (backend, adapter),
                    (evidence / "events.jsonl").open("x", buffering=1) as stream,
                ):

                    def log(record):
                        stream.write(
                            json.dumps(record, default=json_value, allow_nan=False)
                            + "\n"
                        )

                    try:
                        backend.reset(config)
                        ticks, targets = expert_prefix(
                            release / binding["trace"], release / binding["source"]
                        )
                        success = execute_expert(
                            backend,
                            adapter,
                            ticks,
                            targets,
                            binding["expected_initial"]["objects_task"]["target"],
                            log,
                            remaining,
                        )
                        receipt.update(
                            category="diagnostic_complete", success=success, expert=True
                        )
                    finally:
                        try:
                            backend.hold()
                        finally:
                            backend.close(evidence)
            else:
                with open_policy(args.uri, "checkpoint", 90) as policy:
                    persist(
                        args.output / "rng-reset.json",
                        reset_remote(policy, config["policy_rng_seed"], args.episode),
                    )
                    result = run_condition(config, row, factory, policy, evidence)
                receipt.update(
                    category=classify(result),
                    result=json.loads(
                        json.dumps(result, default=json_value, allow_nan=False)
                    ),
                )
    except (Exception, KeyboardInterrupt) as error:
        detail = str(error)
        if isinstance(error, G2Error):
            category = classify(
                dict(
                    outcome="timeout"
                    if error.status == Status.TIMEOUT
                    else "control_error",
                    failure=dict(
                        detail=detail,
                        code=error.status.value,
                        diagnostic=error.diagnostic,
                    ),
                )
            )
            if error.status in (
                Status.PLAN_FAILED,
                Status.IK_FAILED,
                Status.UNREACHABLE,
            ):
                category = "diagnostic_complete"
        else:
            category = "infrastructure_failure"
        if args.probe and isinstance(error, InitialMismatch):
            category = (
                "actual_safety_stop" if "geometry blocked" in detail else "not_admitted"
            )
        receipt.update(category=category, detail=detail, success=False)
        if isinstance(error, G2Error):
            receipt["diagnostic"] = json.loads(
                json.dumps(error.diagnostic, default=json_value, allow_nan=False)
            )
    finally:
        if cleanup_failures:
            receipt.update(
                category="infrastructure_failure", cleanup_failures=cleanup_failures
            )
        receipt["elapsed_s"] = time.monotonic() - started
        persist(args.output / "receipt.json", receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "pace", "output", "model-startup"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("episode", "uri"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--probe", action="store_true")
    run(p.parse_args())
