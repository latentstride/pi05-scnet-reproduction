"""Build reviewable G2-E release from the immutable previous release, without submission."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(project, output):
    evidence = project / "research/g2/g2e-20261106-r1"
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(evidence / "base-release.zip") as z:
        z.extractall(output)
    (output / "SHA256SUMS").unlink()
    shutil.rmtree(output / "configs")
    (output / "configs").mkdir()
    if (output / "expert").exists():
        shutil.rmtree(output / "expert")
    for path in (output / "scripts").iterdir():
        current = project / "scripts" / path.name
        if current.is_file():
            shutil.copy2(current, path)
    for path in (project / "scripts").glob("g2e_*.py"):
        shutil.copy2(path, output / "scripts" / path.name)
    original = project / "research/g2/failure-audit-127054052-r1/release-context"
    old = json.loads((original / "run_lock.json").read_text())
    rows = []
    bindings = {}
    # Counterbalanced order; each query RNG stream resets within environment seed.
    order = [(0, 1), (1, 31), (0, 26), (1, 4), (0, 4), (1, 26), (0, 31), (1, 1)]
    for seed, window in order:
        base = f"seed{seed}-A-native-w1"
        episode = f"g2e-seed{seed}-native-w{window}"
        config = json.loads((original / "configs" / f"{base}.json").read_text())
        config.update(
            episode_id=episode,
            condition=f"E-native-w{window}",
            window=window,
            max_actions=300,
            timeout_s=1150,
            execution_protocol="g2e_native_guarded_prefix_v1",
            policy_rng_seed=20261106 + seed,
            formal_release_ready=False,
            contact_mode="g2e_native_measured_contact_v1",
            budget_scope="common 1150s cooperative / 1200s process / 25000 physics-step ceilings",
        )
        path = output / "configs" / f"{episode}.json"
        write(path, config)
        row = deepcopy(next(r for r in old["episodes"] if r["episode_id"] == base))
        row.update(episode_id=episode, comparison_id="native-window-timing")
        stable = {
            k: v
            for k, v in config.items()
            if k not in ("reset_fixture", "scene_seed", "episode_id", "policy_rng_seed")
        }
        row["condition"].update(
            id=config["condition"],
            window_actions=window,
            config_sha256=sha256(
                json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        rows.append(row)
        binding = deepcopy(old["bindings"][base])
        binding["config_sha256"] = digest(path)
        bindings[episode] = binding
    write(
        output / "run_lock.json",
        dict(
            schema_version=1,
            status="frozen",
            episodes=rows,
            bindings=bindings,
            artifacts=old["artifacts"],
        ),
    )
    spec = json.loads((original / "launch-spec.json").read_text())
    spec.update(
        stage_timeout_s=dict(dynamic=1900, aloha=9600, statistics=120),
        supervisor_timeout_s=11620,
        graphics_timeout_s=300,
        approved_allocation_s=12600,
        trial_timeout_s={f"E-native-w{k}": 1150 for k in (1, 4, 26, 31)},
        expert_scope="no ARX or expert stages in native timing experiment",
        approval_status="awaiting_user_approval",
        pilot_trials=8,
        proposed_budget_at_historical_rate_yuan=12.6,
    )
    write(output / "launch-spec.json", spec)
    entry = (
        (project / "scripts/pace_g2.slurm")
        .read_text()
        .replace("05:00:00", "03:30:00")
        .replace("pace-g2-%j", "pace-g2e-%j")
        .replace("results/pace-g2-", "results/pace-g2e-")
        .replace("g2_single_launch.py", "g2e_launch.py")
        .replace("g2_host_supervisor.py", "g2e_host_supervisor.py")
        .replace("run-admitted-g2-plan.sh", "run-admitted-g2e-plan.sh")
    )
    (output / "scripts/pace_g2e.slurm").write_text(entry)
    wrapper = (
        (project / "scripts/run-admitted-g2-plan.sh")
        .read_text()
        .replace("g2_host_supervisor.py", "g2e_host_supervisor.py")
        .replace("r.get('complete')", "r.get('execution_complete')")
    )
    (output / "scripts/run-admitted-g2e-plan.sh").write_text(wrapper)
    write(
        output / "release-admission.json",
        dict(
            status="awaiting_user_approval",
            gpu_release_ready=False,
            cpu_release_validated=False,
            formal_score_eligible=False,
            approved_allocation_s=12600,
            approved_node="gnode39",
            run_lock_sha256=digest(output / "run_lock.json"),
            launch_spec_sha256=digest(output / "launch-spec.json"),
        ),
    )
    manifest(output)


def manifest(output):
    files = sorted(
        p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest(p)}  {p.relative_to(output)}\n" for p in files)
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("project", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    prepare(a.project.resolve(), a.output.resolve())
