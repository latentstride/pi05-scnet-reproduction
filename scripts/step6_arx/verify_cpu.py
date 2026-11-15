"""Verify CPU acceptance from captured evidence, including serialized WebSocket payloads."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .analyze import summarize
from .protocol import actions_from, decode, validate_request


def verify(root):
    summary = summarize(root)
    if not summary["fixture"]:
        raise ValueError("expected CPU fixture evidence")
    receipts = []
    for pair in summary["pairs"]:
        if pair["matched"] is not True:
            raise ValueError(f'initial readbacks did not match for seed {pair["seed"]}')
        for variant in ("frozen", "lora"):
            trial = root / f'seed-{pair["seed"]:02d}-{variant}'
            r = json.loads((trial / "receipt.json").read_text())
            if r["cleanup_errors"] or r["queries"] < 2 or r["physics_ticks"] <= 0:
                raise ValueError(f"incomplete physical rollout: {trial.name}")
            events = [json.loads(line) for line in (trial / "trajectory.jsonl").read_text().splitlines()]
            queries, responses = [], []
            for event in events:
                wire = event.get("payload")
                if not wire:
                    continue
                binary = (trial / wire["file"]).read_bytes()
                if len(binary) != wire["nbytes"] or sha256(binary).hexdigest() != wire["sha256"]:
                    raise ValueError(f"invalid wire artifact: {wire['file']}")
                message = decode(binary)
                if event["kind"] == "policy_request":
                    validate_request(message, len(queries))
                    if not np.array_equal(message["state"], event["command"]):
                        raise ValueError("wire command differs from physical readback")
                    queries.append(event)
                elif event["kind"] == "policy_response":
                    actions_from(message)
                    responses.append(event)
            physical = [e for e in events if e["kind"] == "physics"]
            if (len(queries) != len(responses) or queries[0]["stamp"]["tick"] != 0
                    or any(a["stamp"]["tick"] >= b["stamp"]["tick"] for a, b in zip(queries, queries[1:]))
                    or any(a["stamp"] != b["stamp"] for a, b in zip(queries, responses))
                    or [e["tick"] for e in physical] != list(range(r["physics_ticks"] + 1))
                    or np.array_equal(physical[0]["actual_qpos"], physical[-1]["actual_qpos"])
                    or np.array_equal(queries[0]["command"], queries[1]["command"])):
                raise ValueError(f"stale, unmatched or incomplete state/requests in {trial.name}")
            if r["outcome"] == "SAFETY_STOP":
                stop = next(e for e in events if e["kind"] == "stop")
                attempted = next(e for e in reversed(events) if e["kind"] == "row_attempt")
                if (r["failure"]["status"] != "position_limit" or stop["tick"] != attempted["tick"]
                        or r["policy_rows_attempted"] != r["policy_actions"] + 1):
                    raise ValueError("the illegal target did not stop in the native joint gate before physics")
            elif r["outcome"] != "ACTION_LIMIT" or r["policy_actions"] != 8:
                raise ValueError("normal CPU fixture did not complete the expected budget")
            receipts.append(dict(episode=r["episode"], queries=r["queries"], ticks=r["physics_ticks"],
                                 rows=r["policy_actions"], outcome=r["outcome"], rgb_status=r["rgb_status"]))
    if not {"ACTION_LIMIT", "SAFETY_STOP"} <= {r["outcome"] for r in receipts}:
        raise ValueError("need both normal and rejected episodes")
    if any(g["n"] for g in summary["formal_groups"].values()):
        raise ValueError("fixture entered formal denominator")
    return dict(trials=receipts, formal_denominator=0, matched_pairs=len(summary["pairs"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.input), indent=2))


if __name__ == "__main__":
    main()
