"""Extract a public, reproducible diagnosis from job 127510363's original logs."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    h = sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_world(path):
    checks = 0
    blocked_pairs = Counter()
    recent = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            checks += 1
            pairs = row["blocked"]
            if pairs:
                for pair in pairs:
                    blocked_pairs[
                        " / ".join(sorted((pair["link_name1"], pair["link_name2"])))
                    ] += 1
                recent.append({"call": row["call"], "blocked": pairs})
    return dict(checks=checks, blocked_pairs=blocked_pairs, blocked_checks=recent)


def collect(run, submission):
    episode = submission["episode"]
    trial = Path("trials") / episode
    folder = trial / "episode"
    source = {}

    def artifact(name):
        path = run / name
        source[str(name)] = file_sha(path)
        return path

    auth = read(artifact("authorization-receipt.json"))
    require(
        all(
            auth[k] == submission[k]
            for k in ("job_id", "archive_sha256", "configuration_sha256", "commit")
        )
        and auth["candidate_sha256"] == submission["manifest_sha256"],
        "authorized deployment identity differs",
    )
    campaign = read(artifact("supervisor/campaign.json"))
    summary = read(artifact("standalone-result.json"))
    receipt = read(artifact(trial / "receipt.json"))
    original = read(artifact(folder / "result.json"))
    probe = read(artifact(Path("startup") / episode / "receipt.json"))
    probe_state = campaign["startup"][episode]
    row = campaign["rows"][episode]
    require(
        len(campaign["rows"]) == 1
        and summary["row"] == row
        and row["result"] == receipt
        and original == receipt["result"]
        and all(probe_state[k] == v for k, v in probe.items()),
        "final receipts disagree",
    )
    model = read(artifact("model/startup.json"))
    graphics = artifact("graphics/check_status.tsv")
    artifact("model/phases.jsonl")
    artifact(folder / "initial-verification.json")
    query, counters, phase_counts, holds, tracked = (
        0,
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    exchanges, barriers = [], []
    with artifact(folder / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["kind"]
            counters[kind] += 1
            if kind == "policy_request":
                query += 1
                exchanges.append(
                    dict(
                        query=query,
                        stamp=event["stamp"],
                        state=event["state"],
                        physical_steps=0,
                        plan_approved=False,
                    )
                )
            elif kind == "source_observability_barrier":
                barriers.append(dict(query=query, **event))
            elif kind == "approved_plan":
                exchanges[-1]["plan_approved"] = True
            elif kind == "executed_step":
                contact = event["contact_state"]
                phase_counts[contact["phase"]] += 1
                holds.update(contact["holders"])
                for side, fingers in contact["contact_fingers"].items():
                    tracked[side] += bool(fingers)
                exchanges[-1]["physical_steps"] += event["physics_step_count"]
    require(
        query
        == original["queries"]
        == counters["policy_response"]
        == counters["policy_validated"]
        and sum(x["plan_approved"] for x in exchanges) == counters["approved_plan"],
        "event and result counters differ",
    )
    terminal_chunks = []
    for exchange in exchanges[-2:]:
        number = exchange["query"]
        actions = np.load(
            artifact(folder / f"action-{number:05d}.npy"), allow_pickle=False
        )
        require(
            actions.shape == (50, 14) and np.isfinite(actions).all(),
            "unexpected model chunk",
        )
        grips = actions[:, [6, 13]]
        crossings = [np.flatnonzero(values <= 0.5) for values in grips.T]
        terminal_chunks.append(
            dict(
                query=number,
                shape=list(actions.shape),
                first_four_grippers=grips[:4].tolist(),
                first_at_or_below_0_5_one_based={
                    side: int(idx[0]) + 1 if len(idx) else None
                    for side, idx in zip(("left", "right"), crossings)
                },
            )
        )
    target = audit_world(artifact(folder / "collision-checks.jsonl"))
    proxy = audit_world(artifact(folder / "proxy-collision-checks.jsonl"))
    return {
        "job_id": auth["job_id"],
        "frozen_deployment": {
            k: auth[k]
            for k in (
                "commit",
                "archive_sha256",
                "candidate_sha256",
                "configuration_sha256",
            )
        },
        "graphics_checks": dict(
            line.split("\t") for line in graphics.read_text().splitlines()
        ),
        "model": {k: model[k] for k in ("checkpoint_revision", "status", "devices")},
        "model_server_starts": campaign["model_server_starts"],
        "peak_gpu_mib": campaign["observed_gpu_peak_mib"],
        "startup_probe": dict(
            probe, process_group_reaped=probe_state["process_group_reaped"]
        ),
        "trial_status": row["status"],
        "category": receipt["category"],
        "campaign_stop_reason": campaign["stop_reason"],
        "process_group_reaped": row["process_group_reaped"],
        "cleanup": receipt["cleanup"],
        "fresh_reset_verified": receipt["fresh_reset_verified"],
        "trial_receipt_elapsed_s": receipt["elapsed_s"],
        "trial": {k: v for k, v in original.items() if k != "video"},
        "event_counts": counters,
        "physical_steps": sum(x["physical_steps"] for x in exchanges),
        "contact_phase_ticks": phase_counts,
        "holder_ticks": holds,
        "tracked_finger_contact_ticks": tracked,
        "last_three_exchanges": exchanges[-3:],
        "source_prefix_barriers": barriers,
        "terminal_chunks": terminal_chunks,
        "target_collision_audit": target,
        "source_proxy_collision_audit": proxy,
        "redacted_fields": ["trial.video (private absolute path)"],
        "source_sha256": source,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_directory", type=Path)
    p.add_argument("submission_receipt", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            collect(a.run_directory, read(a.submission_receipt)),
            ensure_ascii=False,
            indent=2,
        )
    )
