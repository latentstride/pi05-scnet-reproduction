#!/usr/bin/env python3
"""Build fixed launch commands without invoking Slurm."""
import json
from pathlib import Path
import sys


def build(root, release, preflight, out, job, gpu, sim):
    pace = root / "source/PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
    openpi = root / "source/Openpi-PACE-f4e52e6de4cc487a11585a3a7414d610b2883dae"
    fake = out / "g1_1"
    probe = out / "g1_2"
    probe.mkdir()
    fake.mkdir()
    rollout = out / "g1_3"
    # Keep the fake server's default derived port distinct from the checkpoint server.
    checkpoint_port = 50000 + int(job) % 14000
    uri = "ws://127.0.0.1:{}".format(checkpoint_port)
    site = "{}:{}".format(root / "tools/pace-smoke-deps", openpi / "packages/openpi-client/src")
    sim = sim + ["env", "SLURM_JOB_ID=" + job, "PYTHONFAULTHANDLER=1", "PYTHONPATH=" + site,
                 "MPLBACKEND=Agg"]
    model = ["env", "PI05_ROOT=" + str(root), "bash", str(release / "pace_g1_model_env.sh"), gpu]
    return {
        "job_id": job, "allocated_gpu_uuid": gpu, "release_commit": (release / "RELEASE_COMMIT").read_text().strip(),
        "timeouts": {"G1.1": 720, "model_ready": 780, "G1.2": 420, "G1.3": 720},
        "g1_1": sim + ["bash", str(release / "run_pace_native_smoke.sh"), "--native", str(fake),
                       str(preflight / "simulator")],
        "model_server": model + [str(release / "pace_model_smoke.py"), "--port", str(checkpoint_port),
                                 "--require-gpu", "--openpi-root", str(openpi), "--download-dir",
                                 str(root / "cache/pace-handover"), "--diagnostics-dir", str(probe / "model"),
                                 "--verified-model", str(preflight / "model/startup.json")],
        "ready_line": "checkpoint policy on ws://127.0.0.1:{}".format(checkpoint_port),
        "g1_2_query": sim + ["python", str(release / "pace_g1_query.py"), str(fake / "rollout"), str(probe),
                             "--uri", uri, "--timeout", "390"],
        "g1_3": sim + ["bash", str(release / "pace_g1_rollout.sh"), str(pace), str(fake / "rollout"),
                       str(rollout), uri],
        "monitor": ["nvidia-smi", "--id=" + gpu, "--query-gpu=timestamp,uuid,memory.used,memory.total",
                    "--format=csv,noheader,nounits", "--loop=2"],
        "memory_probe": ["nvidia-smi", "--id=" + gpu, "--query-gpu=memory.used,memory.total",
                         "--format=csv,noheader,nounits"],
        "memory_margin_mib": 1024,
    }


if __name__ == "__main__":
    root, release, preflight, out, job, gpu = sys.argv[1:7]
    print(json.dumps(build(*[Path(x) for x in (root, release, preflight, out)], job, gpu, sys.argv[7:]), indent=2))
