"""Verify offline archives, frozen runtime members and the submission entry point."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNTIME_CHANGED = {
    "scripts/g2_arx_direct.py",
    "scripts/g2_remaining_episode.py",
    "scripts/g2_robotwin_backend.py",
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_candidate(label, seed, old_job):
    folder = HERE / "gpu-candidates" / label
    package = json.loads((folder / "package.json").read_text())
    archive = folder / "g2-standalone-arx.tar.gz"
    assert digest(archive) == package["archive_sha256"]
    with TemporaryDirectory() as temporary, tarfile.open(archive) as candidate:
        candidate.extractall(temporary, filter="data")
        release = (Path(temporary) / "g2-standalone-arx").resolve()
        assert digest(release / "SHA256SUMS") == package["manifest_sha256"]
        assert digest(release / "standalone-review.json") == package["configuration_sha256"]
        sys.path.insert(0, str(release / "scripts"))
        from g2_standalone_deploy import verify_standalone
        plan, selected = verify_standalone(release)
        sys.path.pop(0)
        trial, = [row for row in plan["trials"] if row["episode_id"] == selected["episode"]]
        config = release / "configs" / f"{selected['episode']}.json"
        assert digest(config) == package["trial_config_sha256"]
        assert json.loads(config.read_text())["policy_rng_seed"] == trial["policy_rng_seed"] == seed
        assert (trial["seed"], trial["window"], trial["max_policy_actions"],
                trial["hard_timeout_s"], selected["wall_s"], selected["fee_cap"]) == (
                    1, 50, 800, 2600, 5400, "5.40")
        assert not selected["submission_allowed"] and not selected["automatic_retry"]
        subprocess.run(["bash", "-n", str(release / "scripts/pace_g2_standalone.slurm")],
                       check=True)
        for name in RUNTIME_CHANGED:
            assert digest(release / name) == digest(ROOT / name)
        return dict(package_members_verified=len((release / "SHA256SUMS").read_text().splitlines()),
                    standalone_verifier="PASS", slurm_shell_syntax="PASS", executed_sbatch=False,
                    changed_runtime_members={name: digest(release / name)
                                             for name in sorted(RUNTIME_CHANGED)},
                    planned_policy_seed=seed, scene_seed=trial["seed"], historical_baseline_job=old_job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for label, seed, job in (("G1", 20261107, 127999407), ("G2", 20261108, 127999421)):
        report = verify_candidate(label, seed, job)
        print(label, json.dumps(report, sort_keys=True))
