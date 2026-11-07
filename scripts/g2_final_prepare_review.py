"""Build a local diagnostic candidate from frozen configs, sources and traces."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile

from g2_final_expert import expert_prefix
from g2_remaining_conditions import configure
from g2_remaining_plan import build
from pace_g1_orchestrate import persist

RUNTIME_MODULES = """
g2_adapter g2_assets g2_capture g2_collision g2_contact g2_episode g2_export
g2_final_campaign g2_final_deploy g2_final_expert g2_final_statistics g2_final_trial
g2_finger_force g2_initial g2_job_admission g2_kinematics g2_locked_task g2_motion
g2_native_backend g2_native_expert_cpu g2_orchestrator g2_proxy_collision
g2_remaining_conditions g2_remaining_coordinator g2_remaining_episode
g2_remaining_geometry g2_remaining_router g2_robotwin_backend g2_run_task
g2_single_launch g2_task_runtime g2e_host_supervisor g2e_model_entry
g2e_native_guard g2e_protocol pace_client_smoke pace_diagnostics pace_download
pace_g1_orchestrate pace_model_smoke pace_runtime robotwin_control_geometry
robotwin_gpu_smoke robotwin_graphics_driver summarize_experiments verify_robotwin_fk
""".split()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def archive_candidate(output, destination):
    """Archive exactly the manifest files without operating-system metadata."""
    manifest = output / "SHA256SUMS"
    names = [line.split("  ", 1)[1] for line in manifest.read_text().splitlines()]
    with tarfile.open(destination, "x:gz") as archive:
        for name in ["SHA256SUMS", *names]:
            path = output / name
            if path.is_symlink() or not path.is_file() or ".." in Path(name).parts:
                raise ValueError("invalid candidate archive member")
            archive.add(path, arcname=str(Path(output.name) / name), recursive=False)


def prepare(project, output, *, unified=False):
    output.mkdir(parents=True, exist_ok=False)
    configs = output / "configs"
    configs.mkdir()
    if unified:
        from g2_unified_plan import build as build_unified

        plan = build_unified(project)
    else:
        plan = build(project)
    old = project / "research/g2/failure-audit-127054052-r1/release-context"
    prior = json.loads((old / "run_lock.json").read_text())
    manifest = json.loads((old / "pace_download_manifest.json").read_text())
    plan["checkpoint"] = dict(
        revision=manifest["revision"],
        manifest_sha256=digest(old / "pace_download_manifest.json"),
        norm_stats_sha256=manifest["norm_stats_sha256"],
    )
    bindings = {}
    for row in plan["trials"]:
        episode = row["episode_id"]
        seed = row["seed"]
        condition = "A-native-w1" if row["embodiment"] == "aloha-agilex" else "B-arx-w1"
        key = f"seed{seed}-{condition}"
        base = json.loads((old / "configs" / f"{key}.json").read_text())
        effective = row
        if "expert" in row["method"]:
            effective = dict(row, method="frozen_exact_mapping", max_policy_actions=300)
        path = configs / f"{episode}.json"
        persist(path, configure(base, effective, allow_native_h50=unified))
        binding = dict(
            config_sha256=digest(path),
            expected_initial=prior["bindings"][key]["expected_initial"],
        )
        if effective is not row:
            folder = project / f"research/g2/expert-dense-episode{seed}-20261106"
            checksums = json.loads((folder / "checksums.json").read_text())
            target = output / "expert" / f"seed{seed}"
            target.mkdir(parents=True)
            for field, name in (
                ("trace", "runtime-trace.npz"),
                ("source", "input.json"),
            ):
                source = folder / name
                if digest(source) != checksums[str(source.relative_to(project))]:
                    raise ValueError(f"frozen expert {field} hash mismatch")
                shutil.copyfile(source, target / name)
                binding[field] = str((target / name).relative_to(output))
            expert_prefix(output / binding["trace"], output / binding["source"])
        bindings[episode] = binding
    plan["trial_bindings"] = bindings
    plan["status"] = "CANDIDATE_DIAGNOSTIC_ONLY"
    plan["early_stop"].append(
        "verified same-tick contact stop and full cleanup end only that trial; the next trial repeats independent measured reset and geometry admission"
    )
    quote = (
        project / "research/g2/final-cpu-review/scnet-rate-contact-stop-20261107.json"
    )
    if not quote.is_file():
        quote = quote.with_name("scnet-rate-20261107.json")
    if quote.is_file() and not unified:
        shutil.copyfile(quote, output / "catalog-rate.json")
        plan["budget"].update(
            rate_status="catalog_verified_at_observation",
            rate_evidence_sha256=digest(quote),
        )
    persist(output / "unified-plan.json", plan)
    persist(
        output / "release-admission.json",
        dict(
            submission_allowed=False,
            entrypoint="scripts/g2_final_deploy.py",
            authorization="separate allocation-bound approval",
        ),
    )
    frozen_spec = json.loads((old / "launch-spec.json").read_text())
    persist(
        output / "launch-spec.json",
        {
            key: frozen_spec[key]
            for key in ("model_cpu_receipt", "model_cpu_receipt_sha256")
        },
    )
    for key, name in (
        ("asset_audit", "asset-audit.json"),
        ("source_audit", "source-audit.json"),
    ):
        if digest(old / name) != prior["artifacts"][key]:
            raise ValueError(f"frozen {name} differs from the prior lock")
        shutil.copyfile(old / name, output / name)
    shutil.copyfile(
        old / "pace_download_manifest.json", output / "pace_download_manifest.json"
    )
    shutil.copyfile(
        project / "evaluation_schema.json", output / "evaluation_schema.json"
    )
    (output / "scripts").mkdir()
    runtime_files = [name + ".py" for name in RUNTIME_MODULES] + [
        "g2_graphics.sh",
        "pace_g1_model_env.sh",
        "pace_g2_unified.slurm" if unified else "pace_g2_final.slurm",
    ]
    if unified:
        runtime_files += ["g2_unified_plan.py", "g2_unified_exchange.py"]
    for name in runtime_files:
        shutil.copy2(project / "scripts" / name, output / "scripts" / name)
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{digest(p)}  {p.relative_to(output)}\n"
            for p in sorted(output.rglob("*"))
            if p.is_file()
        )
    )
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--unified-transfer", action="store_true")
    args = parser.parse_args()
    prepare(
        args.project.resolve(), args.output.resolve(), unified=args.unified_transfer
    )
    if args.archive:
        archive_candidate(args.output.resolve(), args.archive.resolve())
