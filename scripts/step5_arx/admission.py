"""Release-bound CPU evidence and one-use signed CUDA exception admission."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile

CPU_VERIFIED = "CPU_VERIFIED"
CPU_NUMERIC_EXCEPTION = "CPU_NUMERIC_EXCEPTION"
CPU_STRUCTURAL_ONLY = "CPU_STRUCTURAL_ONLY"
PENDING = "PENDING"
PARTITION = "ksagnormal01"
GPU_TYPE = "RTX4090-PCIE-24GB-LS"
JOB_NAME = "cpf-step5-arx-validation"
WALLTIME = "06:00:00"
SHORT_WALLTIME = "01:30:00"
TRUST_KEY = ".step5-trust/gpu-approval-ed25519.pem"
CLAIMS = "results/.step5-gpu-claims"


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def canonical_under(root, path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path == root or not path.is_relative_to(root):
        raise ValueError(f"Expected a canonical path below the project root: {path}")
    return path


def release_identity(root, release):
    canonical_under(root, release)
    archive = canonical_under(root, release / "release.tar")
    commit = (release / "COMMIT").read_text().strip()
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Missing full release commit SHA")
    work = release / "scripts/step5_arx"
    files = sorted(work.glob("*.py")) + [
        work / "env.sh", work / "train.slurm", release / "pace_download_manifest.json",
        release / "docs/step5/data_audit.json",
    ]
    with tarfile.open(archive) as package:
        if package.pax_headers.get("comment") != commit:
            raise ValueError("Release archive does not identify the requested Git commit")
        for path in files:
            canonical_under(root, path)
            entry = package.getmember(path.relative_to(release).as_posix())
            if not entry.isfile() or package.extractfile(entry).read() != path.read_bytes():
                raise ValueError(f"Release archive does not match installed file: {path}")
    return dict(commit=commit, archive_sha256=file_sha(archive), path=str(release))


def verify_evidence(root, evidence, status):
    required = {"structure", "cpu_step_one", "cpu_second_step", "fixed_input", "last_leaf"} if (
        status == CPU_NUMERIC_EXCEPTION) else {"structure", "cpu_verified_numeric"}
    if not isinstance(evidence, dict) or not required.issubset(evidence):
        raise ValueError("Missing release admission evidence")
    for name, entry in evidence.items():
        path = canonical_under(root, entry["path"])
        if file_sha(path) != entry["sha256"]:
            raise ValueError(f"Admission evidence digest mismatch: {name}")
    structural = json.loads(Path(evidence["structure"]["path"]).read_text())
    flags = ("full_initial_eval_lowered", "full_eval_and_sampling_traced", "abstract_update_traced",
             "checkpoint_recovery_checked")
    if not all(structural.get(flag) is True for flag in flags):
        raise ValueError("Structural evidence does not cover all required checks")
    return structural


def make_scope(args, token, account, log, walltime=WALLTIME):
    if not isinstance(token, str) or not re.fullmatch(r"[a-f0-9]{32}", token):
        raise ValueError("Invalid one-use Slurm token")
    if not isinstance(account, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", account):
        raise ValueError("Invalid Slurm account")
    log = canonical_under(args.project_root, log)
    if not log.parent.is_dir() or log.name.count("%j") != 1 or log.name.count("%") != 1:
        raise ValueError("Slurm stdout requires an existing directory and one %j job ID placeholder")
    return dict(token=token, account=account, partition=PARTITION, gpu_type=GPU_TYPE, gpus=1, cpus=8,
                memory_gib=64, time_limit=walltime, job_name=JOB_NAME, root=str(args.project_root),
                dataset=str(args.dataset), data_audit=str(args.data_audit),
                run=str(args.run), resume=args.resume, log=str(log))


def claim_path(root, token):
    return canonical_under(root, root / CLAIMS / token)


def verify_approval(root, packet, receipt, raw):
    authorization = packet.get("authorization", {})
    if authorization.get("state") != "APPROVED" or not all(
            isinstance(authorization.get(k), str) and authorization[k].strip()
            for k in ("approved_by", "reference")):
        raise PermissionError("CPU numerical exception has no explicit approval")
    approval_token = canonical_under(root, root / TRUST_KEY)
    signature = canonical_under(root, receipt.with_name(receipt.name + ".sig"))
    if not approval_token.is_file() or not signature.is_file():
        raise PermissionError("CPU numerical exception requires a trusted Ed25519 signature")
    signature_bytes = signature.read_bytes()
    if len(signature_bytes) != 64:
        raise PermissionError("CPU numerical exception signature is invalid")
    # Verify the exact bytes already parsed, even if the source file changes before openssl runs.
    with tempfile.TemporaryDirectory() as temp:
        data, sig = Path(temp) / "packet", Path(temp) / "signature"
        data.write_bytes(raw)
        sig.write_bytes(signature_bytes)
        result = subprocess.run(
            ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(approval_token),
             "-rawin", "-in", str(data), "-sigfile", str(sig)],
            capture_output=True, check=False, timeout=15,
        )
    if result.returncode != 0:
        raise PermissionError("CPU numerical exception signature did not verify")


def verify_scope(args, scope):
    walltime = scope.get("time_limit")
    if walltime not in {WALLTIME, SHORT_WALLTIME} or scope != make_scope(
            args, scope.get("token"), scope.get("account"), scope.get("log"), walltime):
        raise PermissionError("CPU numerical exception does not match the requested job")


def validate_slurm_record(record, job_id, scope, release):
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9]*)=([^\s]+)", record))
    if not re.fullmatch(r"[1-9]\d*", job_id or "") or fields.get("JobId") != job_id:
        raise PermissionError("Slurm job record does not match the allocation")
    expected = dict(JobName=scope["job_name"], Account=scope["account"], Partition=scope["partition"],
                    TimeLimit=scope["time_limit"], NumTasks="1", NumCPUs="8",
                    WorkDir=scope["root"], Command=str(release / "scripts/step5_arx/train.slurm"),
                    StdOut=scope["log"].replace("%j", job_id), Comment="step5-arx:" + scope["token"],
                    MinMemoryNode="64G", Requeue="0")
    if fields.get("NumNodes") not in {"1", "1-1"}:
        raise PermissionError("CPU numerical exception requires one Slurm node")
    if fields.get("TresPerNode") not in {f"gres:gpu:{GPU_TYPE}:1", f"gres/gpu:{GPU_TYPE}:1"}:
        raise PermissionError("CPU numerical exception requires one typed RTX 4090 GPU")
    mismatch = [key for key, value in expected.items() if fields.get(key) != value]
    if mismatch:
        raise PermissionError("Slurm allocation does not match approval fields: " + ", ".join(mismatch))


def create_once(path, record, error_message):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with path.open("x") as handle:
            json.dump({**record, "created_at": datetime.now(timezone.utc).isoformat()}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise PermissionError(error_message) from error


def reservation_path(root, token):
    return claim_path(root, token).with_suffix(".submission")


def reserve_submission(root, scope, receipt_sha):
    path = reservation_path(root, scope["token"])
    create_once(path, dict(receipt_sha256=receipt_sha, run=scope["run"]),
                "CPU numerical exception has already reserved its sole submission")


def verify_reservation(root, scope, receipt_sha):
    path = reservation_path(root, scope["token"])
    if not path.is_file() or path.is_symlink():
        raise PermissionError("CPU numerical exception requires a pre-submit reservation")
    reservation = json.loads(path.read_text())
    if reservation.get("run") != scope["run"] or reservation.get("receipt_sha256") != receipt_sha:
        raise PermissionError("CPU numerical exception reservation does not match the approved receipt")


def consume(root, scope, job_id, receipt_sha):
    create_once(claim_path(root, scope["token"]),
                dict(job_id=job_id, receipt_sha256=receipt_sha, run=scope["run"]),
                "CPU numerical exception has already been consumed")


def submission_command(args, release, scope):
    relative = [str(args.project_root), str(release), str(args.dataset), str(args.data_audit),
                str(args.cpu_receipt), str(args.run)]
    if args.resume:
        relative.append("--resume")
    return shlex.join([
        "sbatch", "--parsable", f"--job-name={scope['job_name']}", f"--partition={PARTITION}",
        f"--account={scope['account']}", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
        f"--gres=gpu:{GPU_TYPE}:1", f"--time={scope['time_limit']}", "--signal=B:USR1@300", "--no-requeue",
        f"--comment=step5-arx:{scope['token']}", f"--chdir={args.project_root}", f"--output={scope['log']}",
        str(release / "scripts/step5_arx/train.slurm"), *relative,
    ])
