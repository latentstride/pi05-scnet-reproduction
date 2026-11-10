"""Validate the independent deployment path without Slurm, CUDA or a checkpoint."""

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from g2_final_campaign import supervise
from g2_final_deploy import commands, digest
from g2_standalone_deploy import (
    ARX,
    EPISODE,
    bind,
    isolated_port,
    output_for,
    verify_standalone,
)
from g2_standalone_package import package

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "research/g2/author-h50-unified-review/g2-author-h50-review.tar.gz"


class StandaloneDeployTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name).resolve()
        cls.releases = {}
        for mode in ("author", "arx"):
            archive = cls.root / f"{mode}.tar.gz"
            package(ROOT, PARENT, archive, mode)
            with tarfile.open(archive) as source:
                source.extractall(cls.root, filter="data")
            cls.releases[mode] = cls.root / f"g2-standalone-{mode}"

    def runtime(self, mode, job, output):
        release = self.releases[mode]
        plan, selected = verify_standalone(release)
        spec = json.loads((release / "launch-spec.json").read_text())
        with patch(
            "g2_final_deploy.digest", return_value=spec["model_cpu_receipt_sha256"]
        ):
            original = commands(
                self.root,
                release,
                output,
                job,
                plan,
                dict(
                    gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
                    sim_argv=["cpu-stub"],
                ),
                selected["wall_s"],
                self.root / f"{mode}-approval.json",
            )
        return bind(plan, original, release, output, selected, job)

    def test_archives_are_frozen_and_standalone_verification_needs_no_site_packages(
        self,
    ):
        for mode, release in self.releases.items():
            for name in (
                "g2_graphics.sh",
                "pace_g1_model_env.sh",
                "pace_g2_standalone.slurm",
            ):
                subprocess.run(
                    ["bash", "-n", str(release / "scripts" / name)], check=True
                )
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-S",
                    "-B",
                    str(release / "scripts/g2_standalone_deploy.py"),
                    str(release),
                    "--verify",
                ],
                text=True,
            )
            self.assertFalse(json.loads(output)["selected"]["submission_allowed"])
            self.assertNotIn(
                "--dependency",
                (release / "scripts/pace_g2_standalone.slurm").read_text(),
            )
        author, arx = self.releases.values()
        self.assertEqual(
            digest(author / "scripts/g2_author_native_h50.py"),
            digest(ROOT / "scripts/g2_author_native_h50.py"),
        )
        self.assertEqual(
            digest(author / f"configs/{ARX}.json"), digest(arx / f"configs/{ARX}.json")
        )

    def test_one_trial_per_job_and_disjoint_service_ports_and_paths(self):
        job = "127408633"
        ports, paths = set(), set()
        for mode, expected, nprobes in (("author", EPISODE, 0), ("arx", ARX, 1)):
            _, selected = verify_standalone(self.releases[mode])
            out = output_for(self.root, selected, job)
            plan, runtime = self.runtime(mode, job, out)
            self.assertEqual([r["episode_id"] for r in plan["trials"]], [expected])
            self.assertEqual(list(runtime["commands"]), [expected])
            self.assertEqual(len(runtime["probes"]), nprobes)
            port = isolated_port(selected, job)
            self.assertIn(str(port), runtime["model"])
            self.assertIn(f"ws://127.0.0.1:{port}", runtime["commands"][expected])
            self.assertIn(f"G2_STANDALONE_CACHE={out / 'cache'}", runtime["model"])
            self.assertNotIn(
                "pace-g2-unified-", " ".join(runtime["commands"][expected])
            )
            ports.add(port)
            paths.add(out)
        self.assertEqual(len(ports), 2)
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(port < 50000 for port in ports))

    def test_real_supervisor_uses_one_service_and_fails_closed(self):
        noop = [sys.executable, "-c", "pass"]
        for mode, episode in (("author", EPISODE), ("arx", ARX)):
            with self.subTest(mode=mode):
                plan, runtime = self.runtime(mode, "1", self.root / mode)
                runtime.update(
                    model=[
                        sys.executable,
                        "-u",
                        "-c",
                        "import time;print('READY');time.sleep(40)",
                    ],
                    ready_marker="READY",
                    model_timeout_s=5,
                    model_phases_file=None,
                    monitor=[
                        sys.executable,
                        "-u",
                        "-c",
                        "import time;print('x,g,1,24000');time.sleep(40)",
                    ],
                )
                runtime["commands"][episode] = [
                    sys.executable,
                    "-c",
                    "raise SystemExit(4)",
                ]
                if mode == "arx":
                    (key,) = runtime["probes"]
                    runtime["probes"] = {key: (episode, noop)}
                    runtime["read_result"] = lambda _, probe: dict(
                        episode_id=episode,
                        category="admitted",
                        physical_admitted=True,
                        fresh_reset_verified=True,
                        cleanup=dict(
                            scene_opened=True,
                            close_complete=True,
                            scene_context_closed=True,
                        ),
                    )
                summary = supervise(
                    plan, runtime, self.root / f"supervisor-{mode}", allocated=True
                )
                self.assertEqual(summary["model_server_starts"], 1)
                self.assertEqual(
                    summary["rows"][episode]["status"], "infrastructure_failure"
                )
                self.assertEqual(summary["stop_reason"], "trial process failure")
                self.assertNotIn("cleanup_failures", summary)


if __name__ == "__main__":
    unittest.main()
