"""Native-only one-allocation G2-E argv plan. This module never launches a job."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from g2_single_launch import PACE, OPENPI


def build(root, release, output, job, graphics, allocation_end):
    spec = json.loads((release / "launch-spec.json").read_text())
    lock = release / "run_lock.json"
    roster = json.loads(lock.read_text())["episodes"]
    digest = sha256(lock.read_bytes()).hexdigest()
    if len(roster) != 8 or {r["condition"]["embodiment"] for r in roster} != {
        "aloha-agilex"
    }:
        raise ValueError("G2-E requires exactly eight native-only planned trials")
    gpu = graphics["gpu_uuid"]
    if not gpu.startswith("GPU-"):
        raise ValueError("allocation GPU UUID required")
    pace = root / "source" / f"PACE-{PACE}"
    openpi = root / "source" / f"Openpi-PACE-{OPENPI}"
    pythonpath = ":".join(
        map(
            str,
            (
                release / "scripts",
                root / "tools/pace-smoke-deps",
                openpi / "packages/openpi-client/src",
            ),
        )
    )
    sim = graphics["sim_argv"] + [
        "env",
        f"SLURM_JOB_ID={job}",
        "SLURM_JOB_NAME=cpf-pace-g2",
        f"PYTHONPATH={pythonpath}",
        "PYTHONFAULTHANDLER=1",
        "MPLBACKEND=Agg",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
    ]
    py = sim + ["python"]
    port = 50000 + int(job) % 14000
    uri = f"ws://127.0.0.1:{port}"
    receipt = Path(spec["model_cpu_receipt"])
    if (
        receipt.exists()
        and sha256(receipt.read_bytes()).hexdigest() != spec["model_cpu_receipt_sha256"]
    ):
        raise ValueError("model CPU receipt drift")
    model_dir = output / "model"
    model = [
        "env",
        f"PI05_ROOT={root}",
        "bash",
        str(release / "scripts/pace_g1_model_env.sh"),
        gpu,
        str(release / "scripts/g2e_model_entry.py"),
        "--port",
        str(port),
        "--require-gpu",
        "--openpi-root",
        str(openpi),
        "--download-dir",
        str(root / "cache/pace-handover"),
        "--diagnostics-dir",
        str(model_dir),
        "--verified-model",
        str(receipt),
    ]

    def trial(row, startup=False):
        episode = row["episode_id"]
        dest = output / ("startup" if startup else "trials") / episode
        command = py + [
            str(release / "scripts/g2_locked_task.py"),
            "--lock",
            str(lock),
            "--lock-sha",
            digest,
            "--episode-id",
            episode,
            "--config",
            str(release / "configs" / f"{episode}.json"),
            "--pace-root",
            str(pace),
            "--source-audit",
            str(release / "source-audit.json"),
            "--asset-audit",
            str(release / "asset-audit.json"),
            "--model-manifest",
            str(release / "pace_download_manifest.json"),
            "--model-startup",
            str(model_dir / "startup.json"),
            "--output",
            str(dest),
            "--uri",
            uri,
            "--policy-timeout-s",
            "390" if startup else "90",
        ]
        if startup:
            command += ["--startup-timeout-s", "480"]
        return dict(episode_id=episode, output=str(dest), argv=command)

    startup = trial(roster[0], True)
    warm = output / "dynamic-commands.json"
    warm.write_text(json.dumps([startup["argv"]], indent=2) + "\n")
    entries = output / "native-trials.json"
    entries.write_text(json.dumps([trial(row) for row in roster], indent=2) + "\n")
    stages = dict(
        dynamic=dict(
            argv=[
                "python3",
                str(release / "scripts/g2_single_stage.py"),
                str(warm),
                "--monitor",
                str(output / "supervisor/gpu-memory.csv"),
            ],
            requires=[],
            needs_model=True,
            timeout_s=spec["stage_timeout_s"]["dynamic"],
        ),
        aloha=dict(
            argv=[
                "python3",
                str(release / "scripts/g2e_batch.py"),
                str(entries),
                str(output / "batch-status.json"),
            ],
            requires=["dynamic"],
            needs_model=True,
            timeout_s=spec["stage_timeout_s"]["aloha"],
        ),
        statistics=dict(
            argv=py
            + [
                str(release / "scripts/g2e_statistics.py"),
                str(lock),
                str(output / "trials"),
                str(output / "statistics"),
            ],
            requires=["dynamic"],
            needs_model=False,
            timeout_s=spec["stage_timeout_s"]["statistics"],
        ),
    )
    return dict(
        data_kind="diagnostic",
        total_timeout_s=spec["supervisor_timeout_s"],
        deadline_epoch=allocation_end - 60,
        evaluation_summary_file=str(output / "statistics/evaluation-summary.json"),
        model_ready_timeout_s=960,
        model_server=model,
        ready_line=f"checkpoint policy on {uri}",
        model_phases_file=str(model_dir / "phases.jsonl"),
        model_phase_timeouts=dict(
            container_startup=180,
            checkpoint_and_tokenizer_sha256=180,
            openpi_import=480,
            jax_device=120,
            orbax_restore_and_policy_create=600,
        ),
        monitor=[
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=timestamp,uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "--loop=1",
        ],
        stages=stages,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("root", "release", "output", "graphics"):
        p.add_argument(n, type=Path)
    p.add_argument("job")
    p.add_argument("allocation_end", type=float)
    a = p.parse_args()
    plan = build(
        a.root,
        a.release,
        a.output,
        a.job,
        json.loads(a.graphics.read_text()),
        a.allocation_end,
    )
    (a.output / "effective-launch-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
