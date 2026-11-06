"""CPU checks for the allocated command path without touching Slurm or CUDA."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from g2_final_campaign import supervise
from g2_final_deploy import allocation_matches, authorization, commands, digest
from g2_final_trial import classify
from g2_remaining_plan import build
from g2_task_runtime import inspect_assets
from pace_g1_orchestrate import persist


PROJECT = Path(__file__).resolve().parents[1]
GPU = "GPU-00000000-0000-0000-0000-000000000001"


class DeployTests(unittest.TestCase):
    def test_native_admission_does_not_load_arx_assets(self):
        path = (
            PROJECT
            / "research/g2/failure-audit-127054052-r1/release-context"
            / "configs/seed0-A-native-w1.json"
        )
        config = json.loads(path.read_text())
        with (
            patch("g2_task_runtime.load_arms") as load,
            patch("g2_task_runtime.audited_arx_self_profile") as arx,
        ):
            inspect_assets(config, PROJECT, path, native_only=True)
        self.assertEqual(
            [call.args[2] for call in load.call_args_list], ["aloha-agilex"]
        )
        arx.assert_not_called()

    def test_cluster_legacy_tres_and_exact_resource_envelope(self):
        fields = dict(
            JobState="RUNNING",
            JobName="cpf-pace-g2",
            NodeList="gnode39",
            Partition="ksagnormal01",
            Requeue="0",
            NumNodes="1",
            NumCPUs="4",
            MinMemoryNode="32G",
            TRES="cpu=4,mem=32G,node=1,billing=4,gres/gpu=1",
        )
        self.assertTrue(allocation_matches(fields))
        self.assertFalse(allocation_matches(dict(fields, AllocTRES="gres/gpu=10")))
        self.assertFalse(allocation_matches(dict(fields, Requeue="1")))

    def test_commands_bind_four_probes_and_every_trial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = root / "receipt.json"
            receipt.write_text("{}")
            persist(
                root / "launch-spec.json",
                dict(
                    model_cpu_receipt="<PROJECT_ROOT>/receipt.json",
                    model_cpu_receipt_sha256=digest(receipt),
                ),
            )
            plan = build(PROJECT)
            runtime = commands(
                root,
                root,
                root,
                "9001",
                plan,
                dict(gpu_uuid=GPU, sim_argv=["simulator"]),
                29400,
                root / "approval.json",
            )
            self.assertEqual(
                set(runtime["probes"]),
                {(r, s) for r in ("aloha-agilex", "ARX-X5") for s in (0, 1)},
            )
            self.assertEqual(
                list(runtime["commands"]), [r["episode_id"] for r in plan["trials"]]
            )
            for episode, args in runtime["commands"].items():
                self.assertEqual(args[args.index("--episode") + 1], episode)
                self.assertNotIn("--probe", args)
            self.assertIn(
                "g2e_model_entry.py", [Path(arg).name for arg in runtime["model"]]
            )

    def test_live_quote_and_candidate_hash_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "approval.json"
            quote = path.with_suffix(".quote")
            quote.write_text("simulated quote")
            manifest = root / "SHA256SUMS"
            manifest.write_text("simulated manifest")
            approval = dict(
                approved=True,
                candidate_sha256=digest(manifest),
                node="gnode39",
                quote_source="SCNet",
                quote_sha256=digest(quote),
                yuan_per_gpu_hour="3.60",
                quote_valid_until=(
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ).isoformat(),
            )
            persist(path, approval)
            self.assertEqual(authorization(root, path, "9001"), approval)
            persist(path, dict(approval, yuan_per_gpu_hour="3.61"))
            with self.assertRaises(PermissionError):
                authorization(root, path, "9001")
            persist(path, approval)
            quote.write_text("different quote")
            with self.assertRaises(PermissionError):
                authorization(root, path, "9001")

    def test_candidate_and_measured_rejection_are_separate(self):
        for detail, expected in [
            ("native drive candidate blocked", "diagnostic_complete"),
            ("measured state has non-permitted contact", "actual_safety_stop"),
        ]:
            self.assertEqual(
                classify(dict(outcome="control_error", failure=dict(detail=detail))),
                expected,
            )

    def test_live_admission_is_independent_but_measured_stop_is_global(self):
        plan = build(PROJECT)
        native, arx, later = plan["trials"][0], plan["trials"][2], plan["trials"][3]
        command = [sys.executable, "-c", "pass"]
        for row in plan["trials"]:
            row["hard_timeout_s"] = 1
        plan["budget"]["bounded_execution_s"] = 14 + sum(
            plan["budget"]["overhead"].values()
        )
        categories = {
            native["episode_id"]: "not_admitted",
            arx["episode_id"]: "admitted",
        }

        def receipt(episode, probe=False):
            return dict(
                episode_id=episode,
                category=categories[episode] if probe else "actual_safety_stop",
            )

        runtime = dict(
            mode="allocated",
            timeout_s=10,
            admission={},
            model_timeout_s=3,
            ready_marker="READY",
            model=[
                sys.executable,
                "-u",
                "-c",
                "import time; print('READY'); time.sleep(10)",
            ],
            monitor=[
                sys.executable,
                "-u",
                "-c",
                "import time; print('time, GPU-mock, 1, 24576'); time.sleep(10)",
            ],
            probes={
                (r["embodiment"], r["seed"]): (r["episode_id"], command)
                for r in (native, arx)
            },
            commands={r["episode_id"]: command for r in (native, arx, later)},
            read_result=receipt,
        )
        with tempfile.TemporaryDirectory() as temp:
            result = supervise(plan, runtime, Path(temp) / "run", allocated=True)
        self.assertEqual(result["rows"][native["episode_id"]]["status"], "not_run")
        self.assertEqual(
            result["rows"][arx["episode_id"]]["result"]["category"],
            "actual_safety_stop",
        )
        self.assertEqual(result["rows"][later["episode_id"]]["status"], "not_run")
        self.assertEqual(result["stop_reason"], "actual_safety_stop")
        self.assertEqual(result["model_server_starts"], 1)
