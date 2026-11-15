"""Admission regression without weights or a scheduler."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.step5_arx import admission as a


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.release = self.root / "tools/release"
        self.args = SimpleNamespace(project_root=self.root, dataset=self.root / "data.h5",
                                    data_audit=self.root / "audit.json", run=self.root / "run",
                                    cpu_receipt=self.root / "approval.json", resume=False)
        self.scope = a.make_scope(self.args, "a" * 32, "test_account", self.root / "slurm-%j.out")

    def test_release_validates_commit_and_runtime_bytes(self):
        commit = "c" * 40
        files = ("scripts/step5_arx/train.py", "scripts/step5_arx/env.sh", "scripts/step5_arx/train.slurm",
                 "pace_download_manifest.json", "docs/step5/data_audit.json")
        for name in files:
            path = self.release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        (self.release / "COMMIT").write_text(commit)
        archive = self.release / "release.tar"
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": commit}) as out:
            for name in files:
                out.add(self.release / name, arcname=name)
        self.assertEqual(a.release_identity(self.root, self.release)["archive_sha256"], a.file_sha(archive))
        (self.release / "COMMIT").write_text("d" * 40)
        with self.assertRaisesRegex(ValueError, "Git commit"):
            a.release_identity(self.root, self.release)
        (self.release / "COMMIT").write_text(commit)
        (self.release / files[0]).write_text("edited")
        with self.assertRaisesRegex(ValueError, "installed file"):
            a.release_identity(self.root, self.release)

    def test_evidence_and_evidence_digest(self):
        roles = ("structure", "cpu_step_one", "cpu_second_step", "fixed_input", "last_leaf")
        structure = dict(full_initial_eval_lowered=True, full_eval_and_sampling_traced=True,
                         abstract_update_traced=True, checkpoint_recovery_checked=True)
        evidence = {}
        for role in roles:
            path = self.root / role
            path.write_text(json.dumps(structure) if role == "structure" else role)
            evidence[role] = dict(path=str(path), sha256=a.file_sha(path))
        self.assertEqual(a.verify_evidence(self.root, evidence, a.CPU_NUMERIC_EXCEPTION), structure)
        with self.assertRaisesRegex(ValueError, "Missing"):
            a.verify_evidence(self.root, evidence, a.CPU_VERIFIED)
        (self.root / "last_leaf").write_text("edited")
        with self.assertRaisesRegex(ValueError, "digest mismatch: last_leaf"):
            a.verify_evidence(self.root, evidence, a.CPU_NUMERIC_EXCEPTION)

    def test_signed_approval(self):
        receipt = self.args.cpu_receipt
        pending = {"authorization": {"state": "PENDING"}}
        with self.assertRaisesRegex(PermissionError, "no explicit approval"):
            a.verify_approval(self.root, pending, receipt, b"")
        approval = {"authorization": {"state": "APPROVED", "approved_by": "fixture", "reference": "test only"},
                    "scope": self.scope}
        receipt.write_text(json.dumps(approval))
        with self.assertRaisesRegex(PermissionError, "trusted Ed25519"):
            a.verify_approval(self.root, approval, receipt, receipt.read_bytes())
        key = self.root / "only-a-test-key"
        public = self.root / a.TRUST_KEY
        public.parent.mkdir()
        subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", key], check=True)
        subprocess.run(["openssl", "pkey", "-in", key, "-pubout", "-out", public], check=True)
        sig = receipt.with_name(receipt.name + ".sig")
        subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", key, "-rawin", "-in", receipt, "-out", sig],
                       check=True)
        a.verify_approval(self.root, approval, receipt, receipt.read_bytes())
        with self.assertRaisesRegex(PermissionError, "did not verify"):
            a.verify_approval(self.root, approval, receipt, receipt.read_bytes() + b" ")

    def test_scope_and_slurm_contract(self):
        a.verify_scope(self.args, self.scope)
        self.args.resume = True
        with self.assertRaisesRegex(PermissionError, "requested job"):
            a.verify_scope(self.args, self.scope)
        self.args.resume = False
        expected = (
            "JobId=123 JobName=cpf-step5-arx-validation Account=test_account Partition=ksagnormal01 "
            "JobState=RUNNING TimeLimit=06:00:00 NumTasks=1 NumCPUs=8 NumNodes=1 MinMemoryNode=64G "
            "Requeue=0 TresPerNode=gres:gpu:RTX4090-PCIE-24GB-LS:1 "
            f"WorkDir={self.root} Command={self.release}/scripts/step5_arx/train.slurm "
            f"StdOut={self.root}/slurm-123.out Comment=step5-arx:{self.scope['token']}"
        )
        a.validate_slurm_record(expected, "123", self.scope, self.release)
        for pair in (("JobId=123", "JobId=124"), ("Requeue=0", "Requeue=1"), ("NumCPUs=8", "NumCPUs=16"),
                     ("64G", "32G"), ("24GB-LS:1", "24GB-LS:2"), ("RUNNING", "PENDING")):
            with self.subTest(pair=pair), self.assertRaises(PermissionError):
                a.validate_slurm_record(expected.replace(*pair), "123", self.scope, self.release)
        command = a.submission_command(self.args, self.release, self.scope)
        for flag in ("--no-requeue", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
                     "--signal=B:USR1@300", "--gres=gpu:RTX4090-PCIE-24GB-LS:1"):
            self.assertIn(flag, command)

    def test_reservation_and_concurrent_claim(self):
        with self.assertRaisesRegex(PermissionError, "pre-submit reservation"):
            a.verify_reservation(self.root, self.scope, "receipt")
        a.reserve_submission(self.root, self.scope, "receipt")
        a.verify_reservation(self.root, self.scope, "receipt")
        with self.assertRaisesRegex(PermissionError, "does not match"):
            a.verify_reservation(self.root, self.scope, "altered")
        with self.assertRaisesRegex(PermissionError, "sole submission"):
            a.reserve_submission(self.root, self.scope, "receipt")

        def consume(job):
            try:
                a.consume(self.root, self.scope, str(job), "receipt")
            except PermissionError:
                return False
            return True

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(consume, range(8))), 1)
