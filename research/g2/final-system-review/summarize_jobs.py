"""Join unchanged published receipts with SHA-tagged original event counts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_bytes())


def build(repo, event_counts):
    research = repo / "research/g2"
    out = {}
    for job in ("127408633", "127510335", "127510363"):
        p = research / f"job-{job}"
        result = load(p / "result.json")
        account = (p / "sacct-excerpt.txt").read_text().splitlines()[0].split("|")
        if account[0] != job:
            raise ValueError("accounting receipt identity mismatch")
        base = dict(
            slurm=dict(
                state=account[1],
                start_shanghai=account[2],
                end_shanghai=account[3],
                elapsed=account[4],
                exit_code=account[5],
            ),
            public_result_sha256=sha256((p / "result.json").read_bytes()).hexdigest(),
            complete_author_evaluation=False,
        )
        if job == "127408633":
            rows = []
            for trial in result["trials"]:
                key = trial["episode_id"]
                record = event_counts.get(key)
                if (record is None) != (trial["status"] == "not_run"):
                    raise ValueError(
                        f"missing or unanticipated physical evidence: {key}"
                    )
                rows.append({**trial, **(record or {})})
            base.update(
                deployment_commit=result["commit"],
                archive_sha256=result["archive_sha256"],
                trials=rows,
                counts=result["counts"],
                campaign_stop=result["stop_reason"],
            )
        elif job == "127510335":
            trial = load(p / "trial-receipt.json")
            base.update(
                deployment_commit=result["runtime_binding"]["commit"],
                archive_sha256=result["runtime_binding"]["archive_sha256"],
                trials=[
                    dict(
                        episode_id=trial["episode_id"],
                        category=trial["category"],
                        outcome=trial["outcome"],
                        success=trial["success"],
                        queries=result["event_counts"]["author_query"],
                        completed_actions=trial["completed_actions"],
                        physics_steps=trial["physics_steps"],
                    )
                ],
            )
        else:
            trial = result["trial"]
            base.update(
                deployment_commit=result["frozen_deployment"]["commit"],
                archive_sha256=result["frozen_deployment"]["archive_sha256"],
                trials=[
                    dict(
                        episode_id=trial["episode_id"],
                        category=result["category"],
                        outcome=trial["outcome"],
                        success=trial["success"],
                        queries=trial["queries"],
                        completed_actions=trial["policy_actions"],
                        physics_steps=result["physical_steps"],
                    )
                ],
            )
        out[job] = base
    return dict(
        task="handover_block",
        checkpoint_revision="17ef545270cf26ed81169893b60f845f80ce6557",
        norm_stats_sha256="8a1f114a8873dfa0788c7f512851ee3063224ef2b8af270421076a2e4e544960",
        pace_revision="fd2e624cc19325cc9b20a620c82d51c0d8e25cc1",
        openpi_pace_revision="f4e52e6de4cc487a11585a3a7414d610b2883dae",
        jobs=out,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repository", type=Path)
    p.add_argument("unified_event_counts", type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.repository, load(a.unified_event_counts)), indent=2))
