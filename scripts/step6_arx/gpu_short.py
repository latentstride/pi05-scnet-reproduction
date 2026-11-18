"""Preflight and supervise one allocated SCNet job for PR 7 GPU short acceptance."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from .checkpoint import CheckpointUnavailable, file_sha, inspect_checkpoint, read_json

SUITE = Path(__file__).resolve().parents[2]
GRAPHICS_TIMEOUT = 300
MODEL_START_TIMEOUT = 1050
TRIAL_TIMEOUT = 620
WALL_SECONDS = 70 * 60
STEP5_RELEASE = "step5-arx-release-c21eec73ed58b1ff4f307f3764c5deb6abf3469f"
STEP5_CODE_SHA = "1c81c04e06d86428b9a72fd9c435e7c5498e6b7cc6c5890ef1a666271be4ba27"
PACE = "PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
OPENPI = "Openpi-PACE-f4e52e6de4cc487a11585a3a7414d610b2883dae"


def publish(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def preflight(root, run):
    release = root / "tools" / STEP5_RELEASE
    dataset = root / "data/step5-arx/arx-drive-target-v1.h5"
    pr5 = SUITE.parent / "pr5"
    audit = pr5 / "research/g2/g2e-20261106-r1/release/asset-audit.json"
    required = [
        release / "scripts/step5_arx/env.sh", root / "containers/openpi-ubuntu",
        root / "research/robotwin/robotwin-cu122/opt/robotwin/logs/BUILD_COMPLETE",
        root / "tools/pace-model-venv/bin/python", root / "tools/pace-smoke-deps/websockets",
        root / "cache/pace-handover/handover_block/params/manifest.ocdbt",
        root / "source" / PACE / "envs/handover_block.py",
        root / "source" / OPENPI / "packages/openpi-client/src/openpi_client/msgpack_numpy.py",
        dataset, audit,
    ]
    manifest = SUITE.parent / "SHA256SUMS"
    subprocess.run(["sha256sum", "--quiet", "-c", str(manifest)], cwd=SUITE.parent, check=True)
    if not run.is_relative_to(root / "results") or run != run.resolve():
        raise ValueError("expected a canonical training run under the project results directory")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("missing runtime files: " + ", ".join(missing))
    plan = read_json(SUITE / "docs/step6/pairs.json")
    training = release / "scripts/step5_arx"
    hashes = [file_sha(p) for p in sorted(training.glob("*.py"))]
    hashes += [file_sha(training / name) for name in ("env.sh", "train.slurm")]
    data_audit = release / "docs/step5/data_audit.json"
    try:
        expected = dict(code_sha256=sha256("".join(hashes).encode()).hexdigest(),
                        data_audit_sha256=file_sha(data_audit),
                        manifest_sha256=file_sha(release / "pace_download_manifest.json"),
                        dataset_sha256=file_sha(dataset))
    except OSError as error:
        raise ValueError("failed to verify the Step 5 release inputs") from error
    if expected["code_sha256"] != STEP5_CODE_SHA or expected["dataset_sha256"] != plan["dataset_sha256"]:
        raise ValueError("fixed Step 5 release or dataset failed integrity checks")
    selected, identity, checkpoint = inspect_checkpoint(run, plan)
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint identity does not match the fixed Step 5 release and dataset")
    return dict(status="CHECKPOINT_METADATA_VERIFIED", selected_step=selected["step"],
                checkpoint=str(checkpoint), release=str(release), dataset=str(dataset),
                data_audit=str(data_audit), pr5=str(pr5), asset_audit=str(audit),
                identity_sha256=file_sha(run / "identity.json"))


@contextmanager
def process(cmd, output, env=None, *, stderr=subprocess.STDOUT):
    with output.open("wb") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=stderr, env=env, start_new_session=True)
        try:
            yield child
        finally:
            # The subprocess can exit before its container descendants do.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
                # The process group still needs cleanup if the parent has exited.
            child.wait(timeout=5)


def run_step(cmd, timeout, output, env=None, *, stderr=subprocess.STDOUT):
    with process(cmd, output, env, stderr=stderr) as child:
        if child.wait(timeout=timeout) != 0:
            raise RuntimeError(f"exit {child.returncode}: {output.name}")


def ready(child, port):
    deadline = time.monotonic() + MODEL_START_TIMEOUT
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"model service exited before handshake: {child.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("model service startup exceeded its budget")


def stop_on_signal(signum, _frame):
    raise InterruptedError(f"received signal {signum}")


def launch(root, run):
    job = os.environ["SLURM_JOB_ID"]
    if not job.isdecimal() or Path(os.environ["SLURM_SUBMIT_DIR"]).resolve() != root:
        raise ValueError("launch from the project root in one allocated Slurm job")
    out = root / "results" / f"step6-gpu-short-{job}"
    out.mkdir()
    result = dict(status="RUNNING", job_id=job, allocation_seconds=WALL_SECONDS, variants={})
    receipt = out / "acceptance.json"
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    signal.signal(signal.SIGUSR1, stop_on_signal)
    try:
        result["phase"] = "preflight"
        publish(receipt, result)
        checks = preflight(root, run)
        publish(out / "preflight.json", checks)

        result["phase"] = "graphics"
        publish(receipt, result)
        graphics = out / "graphics"
        pr5 = Path(checks["pr5"])
        run_step(["bash", str(pr5 / "scripts/g2_graphics.sh"), "--g2", str(pr5), str(graphics)],
                 GRAPHICS_TIMEOUT, out / "graphics.log")
        graphics_launch = read_json(graphics / "sim-launch.json")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=graphics_launch["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
        deps = root / "tools/pace-smoke-deps"
        client = root / "source" / OPENPI / "packages/openpi-client/src"
        sim = [*graphics_launch["sim_argv"], "env", f"PYTHONPATH={SUITE}:{deps}:{client}",
               "python", "-B", "-u"]
        model = ["bash", str(Path(checks["release"]) / "scripts/step5_arx/env.sh"),
                 "gpu", str(root), str(SUITE / "scripts/step6_arx/serve.py")]
        for variant in ("frozen", "lora"):
            result["phase"] = f"{variant}_restore"
            publish(receipt, result)
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            model_cmd = [*model, variant, "--project-root", str(root), "--dataset", checks["dataset"],
                         "--data-audit", checks["data_audit"], "--run", str(run), "--port", str(port)]
            with process(model_cmd, out / f"{variant}-model.log", env) as child:
                ready(child, port)
                result["phase"] = f"{variant}_first_query"
                publish(receipt, result)
                run_step([*sim, "-m", "scripts.step6_arx.evaluate", "--gpu-short", "--seed", "0",
                          "--variant", variant, f"--{variant}-url", f"ws://127.0.0.1:{port}",
                          "--pr5-root", str(pr5), "--pace-root", str(root / "source" / PACE),
                          "--asset-audit", checks["asset_audit"], "--output", str(out / "pair")],
                         TRIAL_TIMEOUT, out / f"{variant}-sim.log")
            result["phase"] = f"{variant}_verify"
            verification = out / f"{variant}-verification.json"
            with verification.with_suffix(".log").open("wb") as log:
                run_step([*sim, "-m", "scripts.step6_arx.verify_gpu_short",
                          str(out / "pair" / f"seed-00-{variant}")], 60, verification, stderr=log)
            result["variants"][variant] = read_json(verification)
        result.update(status="PASS", phase="complete")
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        publish(receipt, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("root", type=Path)
    parser.add_argument("training_run", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "run":
        launch(root, args.training_run)
    else:
        try:
            result = preflight(root, args.training_run)
        except (CheckpointUnavailable, FileNotFoundError) as error:
            print(json.dumps(dict(status="CHECKPOINT_UNAVAILABLE", reason=str(error))))
            raise SystemExit(2) from None
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
