"""New unified candidate contract and physical/request evidence separation."""

from pathlib import Path
import json
import subprocess
import tempfile
import unittest

from g2_final_prepare_review import prepare, digest
from g2_final_deploy import verify
from g2_final_statistics import steps, summarize
from g2e_protocol import PRELOAD_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]


def remanifest(root):
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{digest(p)}  {p.relative_to(root)}\n"
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS"
        )
    )


class PreloadReleaseTests(unittest.TestCase):
    def test_opted_release_preserves_ten_rows_all_arx_configs_and_expert_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            base = root / "baseline"
            new = root / "candidate"
            old = prepare(ROOT, base, unified=True)
            plan = prepare(ROOT, new, unified=True, native_preload=True)
            self.assertEqual(old["trials"], plan["trials"])
            self.assertEqual(old["budget"], plan["budget"])
            verified = verify(new)
            self.assertEqual(verified["native_preload_control"], PRELOAD_PROTOCOL)
            native = 0
            for row in plan["trials"]:
                relative = Path("configs") / (row["episode_id"] + ".json")
                a = json.loads((base / relative).read_text())
                b = json.loads((new / relative).read_text())
                if row["embodiment"] == "aloha-agilex":
                    native += 1
                    self.assertEqual(b.pop("native_preload_control"), PRELOAD_PROTOCOL)
                    self.assertEqual(a, b)
                else:
                    self.assertEqual(
                        (base / relative).read_bytes(), (new / relative).read_bytes()
                    )
            self.assertEqual(native, 6)
            for p in (base / "expert").rglob("*"):
                if p.is_file():
                    self.assertEqual(
                        p.read_bytes(), (new / p.relative_to(base)).read_bytes()
                    )
            # Deployment --verify must not import NumPy, CUDA or simulator.
            completed = subprocess.run(
                [
                    "python3",
                    "-S",
                    "-B",
                    str(new / "scripts/g2_final_deploy.py"),
                    str(new),
                    "--verify",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_combined_candidate_keeps_roster_budget_experts_and_safety_parameters(self):
        from g2e_protocol import APPROACH_REPAIR, PRELOAD_ESTABLISHMENT

        with tempfile.TemporaryDirectory() as temp:
            base, new = (Path(temp).resolve() / n for n in ("baseline", "combined"))
            old = prepare(ROOT, base, unified=True)
            plan = prepare(
                ROOT,
                new,
                unified=True,
                native_preload=True,
                contact_preload=True,
                approach_repair=True,
            )
            self.assertEqual(old["trials"], plan["trials"])
            self.assertEqual(old["budget"], plan["budget"])
            self.assertEqual(
                (plan["allocations_planned"], plan["model_server_starts"]), (1, 1)
            )
            verify(new)
            native_count = repair_count = 0
            for row in plan["trials"]:
                relative = Path("configs") / (row["episode_id"] + ".json")
                before, after = (
                    json.loads((p / relative).read_text()) for p in (base, new)
                )
                if "expert" not in row["method"]:
                    self.assertEqual(
                        after.pop("free_space_approach_repair"), APPROACH_REPAIR
                    )
                    repair_count += 1
                if row["embodiment"] == "aloha-agilex":
                    self.assertEqual(
                        after.pop("native_preload_control"), PRELOAD_PROTOCOL
                    )
                    self.assertEqual(
                        after.pop("native_preload_establishment"), PRELOAD_ESTABLISHMENT
                    )
                    native_count += 1
                self.assertEqual(before, after)
            self.assertEqual((native_count, repair_count), (6, 8))
            for p in (base / "expert").rglob("*"):
                if p.is_file():
                    self.assertEqual(
                        p.read_bytes(), (new / p.relative_to(base)).read_bytes()
                    )

    def test_rehashed_approach_profile_leak_into_expert_is_rejected(self):
        from g2e_protocol import APPROACH_REPAIR

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve() / "combined"
            plan = prepare(ROOT, root, unified=True, approach_repair=True)
            expert = plan["trials"][-1]["episode_id"]
            p = root / "configs" / (expert + ".json")
            cfg = json.loads(p.read_text())
            cfg["free_space_approach_repair"] = APPROACH_REPAIR
            p.write_text(json.dumps(cfg))
            plan["trial_bindings"][expert]["config_sha256"] = digest(p)
            (root / "unified-plan.json").write_text(json.dumps(plan))
            remanifest(root)
            with self.assertRaisesRegex(ValueError, "approach"):
                verify(root)

    def test_missing_runtime_and_rehashed_disabled_trial_are_rejected(self):
        for fault in ("runtime", "config"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve() / "candidate"
                plan = prepare(ROOT, root, unified=True, native_preload=True)
                if fault == "runtime":
                    (root / "scripts/g2_native_preload.py").unlink()
                else:
                    ep = plan["trials"][0]["episode_id"]
                    p = root / "configs" / (ep + ".json")
                    cfg = json.loads(p.read_text())
                    del cfg["native_preload_control"]
                    p.write_text(json.dumps(cfg))
                    plan["trial_bindings"][ep]["config_sha256"] = digest(p)
                    (root / "unified-plan.json").write_text(json.dumps(plan))
                remanifest(root)
                with self.assertRaisesRegex(ValueError, "preload"):
                    verify(root)

    def test_opt_in_cannot_modify_legacy_fourteen_trial_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve() / "candidate"
            with self.assertRaises(ValueError):
                prepare(ROOT, root, native_preload=True)
            self.assertFalse(root.exists())

    def test_statistics_keep_request_and_application_separate_without_task_override(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            episode = "native"
            folder = root / "trials" / episode / "episode"
            folder.mkdir(parents=True)
            q = [0.0] * 14
            q[6] = 1.0
            control = dict(
                kind="native_preload_control",
                protocol=PRELOAD_PROTOCOL,
                tick=0,
                physics_step_count=0,
                control_call_count=0,
                modes=dict(left="inactive", right="holding_preload"),
                requested_finger_drive_m=[[0.045, 0.045], [-0.02, -0.02]],
                applied_finger_drive_m=[[0.045, 0.045], [-0.01, -0.01]],
                requested_finger_velocity_m_s=[[0.0, 0.0], [-0.001, -0.001]],
                applied_finger_velocity_m_s=[[0.0, 0.0], [0.0, 0.0]],
            )
            rows = [
                dict(
                    kind="native_action_request",
                    action_index_in_episode=0,
                    normalized_qpos_request=q,
                ),
                control,
                dict(
                    kind="executed_step",
                    tick=1,
                    physics_step_count=1,
                    control_call_count=1,
                    actual_fingers=[[0.045, 0.045], [0.03, 0.03]],
                ),
                dict(kind="native_contact_sample", tick=1, contacts=[]),
                dict(
                    kind="native_guard_state",
                    tick=1,
                    contact_state=dict(holders=["right"], releasing=[]),
                ),
            ]
            path = folder / "events.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            evidence = steps(path, native_pair=True)
            self.assertEqual(evidence["physics_steps"], 1)
            self.assertEqual(evidence["close_request_indices"]["right"], [0])
            self.assertEqual(
                evidence["native_preload_control"]["position_interventions"]["right"], 1
            )
            self.assertEqual(
                evidence["final_measured"]["contact_state"]["holders"], ["right"]
            )
            self.assertIn("independent", evidence["physical_release_status"])
            plan = dict(
                schema_version=3,
                native_preload_control=PRELOAD_PROTOCOL,
                trials=[
                    dict(
                        episode_id=episode,
                        seed=0,
                        condition="native",
                        embodiment="aloha-agilex",
                    )
                ],
                comparisons=[],
            )
            summary = dict(
                stop_reason=None,
                rows={
                    episode: dict(
                        status="finished", result=dict(result=dict(success=True))
                    )
                },
            )
            result = summarize(plan, summary, root)
            self.assertTrue(result["trials"][0]["task_predicate"])
            self.assertEqual(
                result["trials"][0]["declared_native_preload_control"], PRELOAD_PROTOCOL
            )

    def test_legacy_statistics_do_not_acquire_preload_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "events.jsonl"
            path.write_text(
                json.dumps(dict(physics_step_count=1, control_call_count=1)) + "\n"
            )
            report = steps(path, native_pair=True)
            self.assertNotIn("native_preload_control", report)
            self.assertNotIn("physical_release_status", report)


if __name__ == "__main__":
    unittest.main()
