"""Describe every saved prediction and preselect native windows from fixed coverage rules."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def minimum_window(rows, coverage):
    indices = [
        row["first_close_index"][0]
        for row in rows
        if row["first_close_index"][0] is not None
    ]
    if not indices:
        raise ValueError("cannot select a window without predicted left closure")
    return next(
        window
        for window in range(1, 51)
        if np.mean(np.array(indices) < window) >= coverage
    )


def analyze(source, output):
    raw = source.read_bytes()
    data = json.loads(raw)
    if len(data) != 8 or sum(map(len, data.values())) != 840:
        raise ValueError(
            "expected all eight original trials and 840 saved prediction blocks"
        )
    output.mkdir(parents=True, exist_ok=False)
    histogram = []
    cases = {}
    for name, rows in sorted(data.items()):
        if not rows or any(
            row["shape"] != [50, 14] or not row["finite"] for row in rows
        ):
            raise ValueError("unexpected prediction shape or nonfinite output")
        sides = {}
        for side, column in [("left", 0), ("right", 1)]:
            counts = Counter(row["first_close_index"][column] for row in rows)
            if any(
                i is not None and (type(i) is not int or not 0 <= i < 50)
                for i in counts
            ):
                raise ValueError("first-closure index outside the 50-action horizon")
            for index in [*range(50), None]:
                histogram.append(
                    dict(
                        episode_id=name,
                        side=side,
                        first_close_index=index,
                        count=counts[index],
                        n=len(rows),
                    )
                )
            finite = sum(counts[i] for i in range(50))
            sides[side] = dict(
                blocks=len(rows),
                blocks_with_close=finite,
                blocks_without_close=counts[None],
                retained={
                    str(k): sum(v for i, v in counts.items() if i is not None and i < k)
                    for k in range(1, 51)
                },
                truncated={
                    str(k): sum(
                        v for i, v in counts.items() if i is not None and i >= k
                    )
                    for k in range(1, 51)
                },
            )
        cases[name] = sides
    native = {n: rows for n, rows in data.items() if "A-native" in n}
    choices = {
        str(coverage): max(minimum_window(rows, coverage) for rows in native.values())
        for coverage in (0.5, 0.9)
    }
    selected = sorted(set(choices.values()))
    (output / "selection.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                source_sha256=sha256(raw).hexdigest(),
                scope="predeclared coverage rule on prior exploratory native trials, before the new GPU evaluation",
                closure_definition="first raw left/right gripper target < 0.5; index zero-based",
                selection_rule="smallest K achieving coverage of closure-proposing blocks in every native seed/window cohort",
                minimum_coverage_windows=choices,
                selected_windows=selected,
                control_windows=[1, 4],
                cases=cases,
                all_blocks=840,
                native_blocks=750,
                arx_blocks=90,
                inference="descriptive correlated blocks; two environmental seeds; no independent-replicate confidence intervals",
            ),
            indent=2,
        )
        + "\n"
    )
    with (output / "first-closure-counts.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(histogram[0]))
        writer.writeheader()
        writer.writerows(histogram)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(analyze(args.source, args.output))
