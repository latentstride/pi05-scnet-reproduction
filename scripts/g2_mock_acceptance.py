#!/usr/bin/env python3
"""Persist synthetic G2 episode evidence. This entry point never requests a GPU."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys
import subprocess

import numpy as np

from g2_episode import ExpertWaypoints, run_episode, save_json

# The fixture is deliberately kept with tests to prevent mistaking it for a simulator backend.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from g2_fixtures import fixture
from test_g2_integration import FixedPolicy, MockRobot, spec


def acceptance(output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for condition in ("Aloha", "ARX"):
        for window in (1, 4):
            adapter, qa, qr, bases = fixture()
            backend = MockRobot(adapter, qa, qr, bases)
            config = spec(window, condition)
            name = f"{condition}-window{window}"
            config["episode_id"] = name
            config["condition"] = ("A_" if condition == "Aloha" else "B_") + name
            row = run_episode(config, backend, FixedPolicy(qa), output / name,
                              adapter=adapter if condition == "ARX" else None)
            rows.append(row)
    for fault in ("model", "shape", "task", "reset", "control", "interrupt", "timeout", "planning"):
        adapter, qa, qr, bases = fixture()
        config = spec(1)
        config["episode_id"] = f"failure-{fault}"
        config["condition"] = "B_ARX-window1"
        if fault == "timeout":
            config["timeout_s"] = .06
        if fault == "planning":
            adapter.target_collision = lambda q, g: False
        row = run_episode(config, MockRobot(adapter, qa, qr, bases, fault),
                          FixedPolicy(qa, fault), output / config["episode_id"], adapter=adapter)
        rows.append(row)
    adapter, qa, qr, bases = fixture()
    config = spec(4, "expert-arx")
    config["episode_id"] = "expert-arx"
    expert_rows = np.tile(np.column_stack((qa, [.5, .5])).ravel(), (4, 1))
    rows.append(run_episode(config, MockRobot(adapter, qa, qr, bases), None, output / "expert-arx",
                            adapter=adapter, expert=ExpertWaypoints(expert_rows, "synthetic_expert_qpos")))
    from test_g2_export import identity
    from g2_export import episode_record
    records, export_manifest = [], []
    for row in rows:
        config = dict(episode_id=row["episode_id"], condition=row["condition"], window=row["window"])
        embodiment = "aloha-agilex" if "Aloha" in row["condition"] else "arx-x5"
        contrast = ("synthetic-transfer-w" + str(row["window"])) if row["episode_id"].startswith(("Aloha", "ARX")) else None
        meta = identity(config, embodiment, contrast)
        if row["episode_id"] == "expert-arx":
            meta["condition"].update(action_source="expert_replay", checkpoint=None, expert_representation="expert_qpos")
        directory = output / row["episode_id"]
        records.append(episode_record(directory, meta))
        export_manifest.append(dict(directory=str(directory), identity=meta))
    save_json(output / "evaluation-records.json", records)
    save_json(output / "export-manifest.json", export_manifest)
    subprocess.run([sys.executable, str(Path(__file__).with_name("summarize_experiments.py")),
                    str(output / "evaluation-records.json"), "--data-kind", "synthetic",
                    "--output", str(output / "statistics")], check=True)
    result = dict(data_kind="synthetic", formal_score_eligible=False, gpu_used=False,
                  simulator_run=False, episodes=rows,
                  outcome_counts=dict(Counter(x["outcome"] for x in rows)),
                  note="Mock success exercises control flow; it is not robot task success.")
    expected = {"success": 5, "policy_failed": 2, "task_unsuccessful": 1, "environment_error": 1,
                "control_error": 1, "interrupted": 1, "timeout": 1, "planning_failed": 1}
    result["passed"] = result["outcome_counts"] == expected
    save_json(output / "mock-acceptance.json", result)
    files = [p for p in output.rglob("*") if p.is_file()]
    (output / "SHA256SUMS").write_text("".join(f"{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output)}\n"
                                             for p in sorted(files)))
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if acceptance(args.output) else 1)
