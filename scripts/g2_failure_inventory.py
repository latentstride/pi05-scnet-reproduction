"""Hash an existing G2 run and extract evidence without modifying its files."""

import argparse
from collections import Counter, deque
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import zipfile


def digest(path):
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(raw, output):
    raw, output = raw.resolve(strict=True), output.resolve()
    if output == raw or raw in output.parents:
        raise ValueError("analysis output must be outside the original run")
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in sorted(raw.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink requires manual provenance review: {path}")
        if path.is_file():
            stat = path.stat()
            files[str(path.relative_to(raw))] = dict(
                bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(path)
            )
            if (
                path.stat().st_mtime_ns != stat.st_mtime_ns
                or path.stat().st_size != stat.st_size
            ):
                raise ValueError("original evidence changed while hashing")
    (output / "inventory.json").write_text(
        json.dumps(dict(raw_root=str(raw), files=files), indent=2) + "\n"
    )
    trials = {}
    with zipfile.ZipFile(
        output / "evidence-selection.zip", "w", zipfile.ZIP_DEFLATED
    ) as bundle:
        for directory in sorted((raw / "trials").iterdir()):
            if not directory.is_dir():
                continue
            name = directory.name
            counts = {
                pattern: len(list(directory.glob(pattern)))
                for pattern in (
                    "action-*.npy",
                    "observation-*.npz",
                    "plan-*.npy",
                    "plan-*.npz",
                )
            }
            required = (
                "result.json",
                "events.jsonl",
                "head.mp4",
                "initial-readback.json",
                "setup.json",
                "formal-record.json",
            )
            missing = [item for item in required if not (directory / item).is_file()]
            result = (
                json.loads((directory / "result.json").read_text())
                if (directory / "result.json").is_file()
                else None
            )
            kinds, phases = Counter(), Counter()
            selected, last_steps, samples = [], deque(maxlen=2), []
            error_sum = error_count = 0
            max_error, first_large_error = 0, None
            prior_clock = None
            clock_errors = []
            events_path = directory / "events.jsonl"
            with (
                events_path.open() if events_path.is_file() else nullcontext(())
            ) as events:
                for line_number, line in enumerate(events, 1):
                    row = json.loads(line)
                    kinds[row.get("kind", "adapter_diagnostic")] += 1
                    phases[row["phase"]] += 1
                    if row.get("kind") == "executed_step":
                        last_steps.append(row)
                        clock = row.get("sim_s")
                        if (
                            clock is not None
                            and prior_clock is not None
                            and clock <= prior_clock
                        ):
                            clock_errors.append(line_number)
                        prior_clock = clock
                        errors = row.get("position_error_m", [])
                        error_sum += sum(x * x for x in errors)
                        error_count += len(errors)
                        if errors:
                            max_error = max(max_error, max(errors))
                            if max(errors) > 0.05 and first_large_error is None:
                                first_large_error = dict(line=line_number, event=row)
                        if row.get("tick", 0) % 250 == 0:
                            samples.append(row)
                    else:
                        selected.append(dict(source_line=line_number, **row))
            bundle.writestr(f"{name}/events-selected.json", json.dumps(selected))
            bundle.writestr(
                f"{name}/control-samples.json", json.dumps(samples + list(last_steps))
            )
            actions = sorted(directory.glob("action-*.npy"))
            plans = sorted(directory.glob("plan-*.np*"))
            for path in sorted(set(actions[-2:] + actions[:1] + plans[-2:])):
                bundle.write(path, f"{name}/{path.name}")
            for item in required:
                if item.endswith(".json") and (directory / item).is_file():
                    bundle.write(directory / item, f"{name}/{item}")
            collisions = {}
            for item in ("collision-checks.jsonl", "proxy-collision-checks.jsonl"):
                path = directory / item
                if not path.is_file():
                    collisions[item] = dict(present=False)
                    continue
                tail, pairs, n = deque(maxlen=40), Counter(), 0
                with path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        n += 1
                        tail.append(row)
                        for block in row.get("blocked", []):
                            pairs[
                                "|".join(
                                    str(block.get(k))
                                    for k in (
                                        "collision_type",
                                        "object_name1",
                                        "link_name1",
                                        "object_name2",
                                        "link_name2",
                                    )
                                )
                            ] += 1
                collisions[item] = dict(
                    present=True, calls=n, blocked_pairs=dict(pairs)
                )
                bundle.writestr(f"{name}/{item}-tail.json", json.dumps(list(tail)))
            trials[name] = dict(
                counts=counts,
                missing=missing,
                result=result,
                event_kinds=dict(kinds),
                event_phases=dict(phases),
                nonmonotonic_clock_lines=clock_errors,
                tracking_position_rmse_m=(error_sum / error_count) ** 0.5
                if error_count
                else None,
                tracking_position_max_m=max_error,
                first_tracking_error_over_50mm=first_large_error,
                collision_logs=collisions,
            )
        for folder in ("supervisor", "model"):
            for path in (raw / folder).glob("*.json"):
                bundle.write(path, f"{folder}/{path.name}")
        for path in raw.glob("*launch*.json"):
            bundle.write(path, path.name)
    (output / "trial-audit.json").write_text(json.dumps(trials, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                files=len(files),
                bytes=sum(r["bytes"] for r in files.values()),
                trials=len(trials),
                archive_bytes=(output / "evidence-selection.zip").stat().st_size,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.raw, args.output)
