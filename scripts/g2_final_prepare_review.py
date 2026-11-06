"""Freeze the review roster, validated policy configurations and source hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from g2_remaining_conditions import configure
from g2_remaining_plan import build


def prepare(project, output):
    output.mkdir(parents=True, exist_ok=False)
    configs = output / "configs"
    configs.mkdir()
    plan = build(project)
    old = project / "research/g2/failure-audit-127054052-r1/release-context/configs"

    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    bindings = {}
    for row in plan["trials"]:
        episode = row["episode_id"]
        if "expert" in row["method"]:
            bindings[episode] = dict(
                status="NOT_VERIFIED",
                reason="live expert phase transition not admitted",
            )
            continue
        condition = "A-native-w1" if row["embodiment"] == "aloha-agilex" else "B-arx-w1"
        base = json.loads((old / f"seed{row['seed']}-{condition}.json").read_text())
        cfg = configure(base, row)
        path = configs / f"{episode}.json"
        write(path, cfg)
        bindings[episode] = dict(
            status="CPU_CONFIG_VALIDATED",
            config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    plan["trial_bindings"] = bindings
    write(output / "unified-plan.json", plan)
    write(
        output / "release-admission.json",
        dict(
            status="CPU_PLAN_ONLY_NOT_SUBMISSION_READY",
            submission_allowed=False,
            runtime_argv_bound=False,
            required=[
                "native updated live-contact dynamics",
                "ARX real-scene and proxy admission",
                "expert handover-to-placement transition",
                "live rate and deployed runtime lock",
                "explicit authorization",
            ],
        ),
    )
    (output / "scripts").mkdir()
    for pattern in (
        "g2*.py",
        "pace_runtime.py",
        "pace_g1_orchestrate.py",
        "robotwin_control_geometry.py",
    ):
        for path in sorted((project / "scripts").glob(pattern)):
            shutil.copyfile(path, output / "scripts" / path.name)
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output)}\n"
            for p in sorted(output.rglob("*"))
            if p.is_file()
        )
    )
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.project, args.output)
