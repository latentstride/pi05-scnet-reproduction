"""Compare observed first exchanges, independent of Job ID and episode labels."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from g2_unified_exchange import hash_array  # noqa: E402


def exchange(directory):
    events = directory / "events.jsonl"
    if not events.is_file():
        return {}
    for line in events.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") in ("response", "author_chunk") and row.get("query") == 0:
            rng = row["rng"]
            path = directory / "chunk-00.npy"
            return dict(
                request=rng["request_sha256"],
                rng=rng["before"],
                sampling_key=rng["sample"],
                next_rng=rng["after"],
                chunk=hash_array(np.load(path, allow_pickle=False)),
            )
    return {}


def historical(path):
    data = json.loads(path.read_text())["first_chunk"]
    rng = data["rng"]
    return dict(
        request=rng["request_sha256"],
        rng=rng["before"],
        sampling_key=rng["sample"],
        next_rng=rng["after"],
        chunk=data["sha256"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "run_root", type=Path, help="directory holding G1..G7 trial directories"
    )
    p.add_argument("--historical", type=Path, required=True)
    a = p.parse_args()
    results = []
    for left, right in (
        ("G1", "G4"),
        ("G2", "G5"),
        ("G3", "G6"),
        ("G2", "G7"),
        ("G1", "127510335"),
    ):
        first = exchange(a.run_root / left)
        second = (
            historical(a.historical)
            if right == "127510335"
            else exchange(a.run_root / right)
        )
        different = sorted(k for k in first if first[k] != second.get(k))
        results.append(
            dict(
                pair=[left, right],
                first_exchange="NOT_REACHED"
                if not first or not second
                else ("DIFFERENT" if different else "MATCH"),
                differences=different if first and second else [],
                left=first,
                right=second,
            )
        )
    print(
        json.dumps(
            dict(
                historical_sha256=sha256(a.historical.read_bytes()).hexdigest(),
                pairs=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
