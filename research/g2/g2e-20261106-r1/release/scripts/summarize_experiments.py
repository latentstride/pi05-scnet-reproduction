#!/usr/bin/env python3
"""Validate episode records and write separate descriptive and paired summaries."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

SCHEMA = Path(__file__).resolve().parents[1] / "evaluation_schema.json"
POSITION_TOL_M = 0.001
ORIENTATION_TOL_DEG = 0.1
JOINT_TOL_RAD = 0.001
GRIPPER_TOL = 0.01


def unique_fields(items):
    fields = {}
    for key, value in items:
        if key in fields:
            raise ValueError(f"duplicate JSON key {key!r}")
        fields[key] = value
    return fields


def finite_float(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number {raw}")
    return value


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_fields, parse_float=finite_float,
                      parse_constant=finite_float)


def read_records(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".json":
            rows = decode(handle.read())
            if not isinstance(rows, list):
                raise ValueError(f"{path}: expected a JSON array of episode records")
            yield from rows
        elif path.suffix.lower() == ".csv":
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            if len(header) != len(fields) or set(header) != fields:
                raise ValueError(f"{path}: CSV header must contain every schema field once")
            for row in reader:
                if None in row or None in row.values():
                    raise ValueError(f"{path}:{reader.line_num}: CSV row width mismatch")
                yield {key: decode(value) for key, value in row.items()}
        else:
            raise ValueError(f"{path}: expected .json or .csv")


def orientation_distance_deg(a, b):
    # Inputs have already passed the unit-norm check. The absolute dot
    # handles the double cover, where q and -q describe the same rotation.
    dot = abs(sum(x * y for x, y in zip(a, b))) / (math.hypot(*a) * math.hypot(*b))
    return math.degrees(2 * math.acos(min(1.0, dot)))


def pose_matches(a, b):
    return (math.dist(a[:3], b[:3]) <= POSITION_TOL_M
            and orientation_distance_deg(a[3:], b[3:]) <= ORIENTATION_TOL_DEG)


def validate_episode(row):
    initial = row["initial"]
    if initial is not None:
        poses = [initial["task_frame_world"], *initial["robot_base_world"].values(),
                 *initial["objects_task"].values()]
        if any(abs(math.hypot(*pose[3:]) - 1.0) > 1e-6 for pose in poses):
            raise ValueError("initial poses require unit WXYZ quaternions")
        if row["match_status"] == "matched" and not initial["common_workspace_pass"]:
            raise ValueError("matched initial state requires a passed common workspace check")
    metrics = row["metrics"]
    for name, numerator, denominator in (
        ("planning", "failures", "attempts"), ("limits", "violations", "checks"),
        ("gripper", "state_mismatches", "samples"), ("actions", "executed", "requested"),
    ):
        metric = metrics[name]
        if metric is not None and metric[numerator] > metric[denominator]:
            raise ValueError(f"{name}.{numerator} exceeds {denominator}")
    for name in ("inference", "control"):
        metric = metrics[name]
        if metric is not None and not metric["calls"] and metric["wall_s"]:
            raise ValueError(f"{name}.wall_s requires at least one call")
    actions = metrics["actions"]
    if actions is not None and actions["requested"] > actions["windows"] * row["condition"]["window_actions"]:
        raise ValueError("requested actions exceed the configured window capacity")


def load_records(paths, data_kind):
    schema = decode(SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    seen = set()
    fixed = {}
    episodes = []
    for path in paths:
        for number, row in enumerate(read_records(path, set(schema["properties"])), 1):
            try:
                validator.validate(row)
                validate_episode(row)
                if row["data_kind"] != data_kind:
                    raise ValueError(f"expected data_kind={data_kind}; found {row['data_kind']}")
                lock = row["run_lock_sha256"]
                identity = lock, row["episode_id"]
                if identity in seen:
                    raise ValueError(f"duplicate episode {identity}")
                seen.add(identity)
                key = lock, row["task"], row["condition"]["id"]
                config = row["condition"], row["environment"]
                if fixed.setdefault(key, config) != config:
                    raise ValueError(f"condition or environment drift under {key}")
            except ValidationError as error:
                field = ".".join(map(str, error.absolute_path)) or "(record)"
                raise ValueError(f"{path} record {number} {field}: {error.message}") from error
            except ValueError as error:
                raise ValueError(f"{path} record {number}: {error}") from error
            episodes.append(row)
    if not episodes:
        raise ValueError("no episode records")
    return episodes


LOCK_FIELDS = ("episode_id", "task", "seed", "comparison_id", "pair_id", "condition", "environment")


def check_formal_roster(path, episodes):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    lock = decode(raw.decode("utf-8"))
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ValueError("invalid run lock schema_version")
    plan = lock.get("episodes")
    if not isinstance(plan, list) or not plan:
        raise ValueError("run lock requires a nonempty episodes roster")
    scheduled = {}
    for item in plan:
        if (not isinstance(item, dict) or set(item) != set(LOCK_FIELDS)
                or not isinstance(item["episode_id"], str)):
            raise ValueError("invalid run lock roster item")
        identity = item["episode_id"]
        if identity in scheduled:
            raise ValueError(f"duplicate planned episode {identity}")
        scheduled[identity] = item

    for row in episodes:
        identity = row["episode_id"]
        if row["run_lock_sha256"] != digest:
            raise ValueError(f"episode {identity}: run lock SHA-256 mismatch")
        if scheduled.get(identity) != {field: row[field] for field in LOCK_FIELDS}:
            raise ValueError(f"episode {identity}: does not match the locked roster")
        del scheduled[identity]
    if scheduled:
        raise ValueError(f"missing {len(scheduled)} locked episodes (first: {next(iter(scheduled))})")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metric_totals(episodes, name, *fields):
    values = [row["metrics"][name] for row in episodes if row["metrics"][name] is not None]
    return [len(values), *(sum(value[field] for value in values) for field in fields)]


def stats_row(key, episodes):
    lock, task, comparison_id, condition_id, reference_id = key
    condition = episodes[0]["condition"]
    outcomes = Counter(row["outcome"] for row in episodes)
    result = dict(run_lock_sha256=lock, task=task, comparison_id=comparison_id or None, condition_id=condition_id,
                  action_source=condition["action_source"], embodiment=condition["embodiment"],
                  expert_representation=condition["expert_representation"],
                  execution=condition["execution"], window_actions=condition["window_actions"],
                  n_episodes=len(episodes), success_rate=outcomes["success"] / len(episodes),
                  trajectory_reference_id=reference_id)
    result.update({f"outcome_{outcome}": outcomes[outcome] for outcome in OUTCOMES})
    for name, numerator, denominator in (
        ("planning", "failures", "attempts"), ("limits", "violations", "checks"),
        ("gripper", "state_mismatches", "samples"), ("inference", "wall_s", "calls"),
        ("control", "wall_s", "calls"), ("actions", "executed", "windows"),
    ):
        count, total, base = metric_totals(episodes, name, numerator, denominator)
        result.update({f"{name}_episodes": count, f"{name}_{numerator}": total,
                       f"{name}_{denominator}": base, f"{name}_{numerator}_per_{denominator}": ratio(total, base)})
    _, requested, elapsed = metric_totals(episodes, "actions", "requested", "sim_elapsed_s")
    result.update(actions_requested=requested, actions_sim_elapsed_s=elapsed,
                  actions_executed_per_requested=ratio(result["actions_executed"], requested))
    for side in ("left", "right"):
        field = f"final_{side}_actual"
        count, total = metric_totals(episodes, "gripper", field)
        result[f"mean_gripper_{field}"] = ratio(total, count)
    trajectories = [row["metrics"]["trajectory"] for row in episodes if row["metrics"]["trajectory"] is not None]
    result.update(trajectory_episodes=len(trajectories), trajectory_samples=sum(t["n"] for t in trajectories))
    for name in ("position_rmse_m", "position_p95_m", "orientation_mean_deg", "orientation_p95_deg"):
        result[f"mean_episode_{name}"] = mean(t[name] for t in trajectories) if trajectories else None
    return result


def condition_tables(episodes):
    groups = defaultdict(list)
    for row in episodes:
        groups[row["run_lock_sha256"], row["task"], row["comparison_id"] or "", row["condition"]["id"]].append(row)
    for key, rows in sorted(groups.items()):
        references = {t["reference_id"] for row in rows if (t := row["metrics"]["trajectory"]) is not None}
        if len(references) > 1:
            raise ValueError(f"mixed trajectory reference definitions under {key}")
        yield stats_row((*key, next(iter(references), None)), rows)


def same_initial(a, b, same_robot):
    objects = a["objects_task"]
    other_objects = b["objects_task"]
    if objects.keys() != other_objects.keys() or not all(
        pose_matches(pose, other_objects[name]) for name, pose in objects.items()
    ):
        return False
    if not same_robot:
        return True
    return (a["reset_fixture_sha256"] == b["reset_fixture_sha256"]
            and pose_matches(a["task_frame_world"], b["task_frame_world"])
            and all(abs(a["actual_gripper"][side] - b["actual_gripper"][side]) <= GRIPPER_TOL
                    and pose_matches(a["robot_base_world"][side], b["robot_base_world"][side])
                    and max(abs(x - y) for x, y in zip(a["actual_arm_qpos_rad"][side],
                                                     b["actual_arm_qpos_rad"][side])) <= JOINT_TOL_RAD
                    for side in ("left", "right")))


def paired_tables(episodes):
    comparisons = defaultdict(lambda: defaultdict(dict))
    for row in episodes:
        if row["comparison_id"] is None:
            continue
        key = row["run_lock_sha256"], row["task"], row["comparison_id"]
        pair = comparisons[key][row["pair_id"]]
        cid = row["condition"]["id"]
        if cid in pair:
            raise ValueError(f"{key} pair {row['pair_id']}: repeated condition {cid}")
        pair[cid] = row

    pair_rows = []
    summaries = []
    for key, pairs in sorted(comparisons.items()):
        by_condition = {cid: row for pair in pairs.values() for cid, row in pair.items()}
        conditions = sorted(by_condition)
        if len(conditions) > 2:
            raise ValueError(f"{key}: a contrast supports exactly two conditions")
        first, second = (conditions + [None])[:2]
        prefix = dict(zip(("run_lock_sha256", "task", "comparison_id"), key))
        prefix.update(condition_a=first, condition_b=second)
        if second is not None:
            a, b = by_condition[first], by_condition[second]
            ca, cb = a["condition"], b["condition"]
            expert_a = ca["action_source"] == "expert_replay"
            expert_b = cb["action_source"] == "expert_replay"
            if expert_a != expert_b or ca["checkpoint"] != cb["checkpoint"]:
                raise ValueError(f"{key}: paired contrasts require matching policy weights and stats or two expert conditions")
            if not expert_a and ca["embodiment"] != cb["embodiment"]:
                if (ca["execution"], ca["window_actions"]) != (cb["execution"], cb["window_actions"]):
                    raise ValueError(f"{key}: cross-embodiment policy conditions differ in execution or window")
            for field in ("robotwin_revision", "pace_revision", "assets_revision", "scene_config_sha256"):
                if a["environment"][field] != b["environment"][field]:
                    raise ValueError(f"{key}: paired conditions differ in {field}")

        coverage = Counter()
        deltas = []
        for pair_id, pair in sorted(pairs.items()):
            a, b = pair.get(first), pair.get(second)
            delta = None
            if a is None or b is None:
                status = "missing_condition"
            elif a["match_status"] != "matched" or b["match_status"] != "matched":
                status = "not_admitted"
            elif not same_initial(a["initial"], b["initial"],
                                  a["condition"]["embodiment"] == b["condition"]["embodiment"]):
                raise ValueError(f"{key} pair {pair_id}: initial states marked matched differ beyond tolerance")
            else:
                status = "matched"
                delta = int(b["outcome"] == "success") - int(a["outcome"] == "success")
                deltas.append(delta)
            coverage[status] += 1
            pair_rows.append(dict(prefix, pair_id=pair_id, status=status, success_delta_b_minus_a=delta,
                                  episode_a=a["episode_id"] if a else None,
                                  episode_b=b["episode_id"] if b else None))
        summaries.append(dict(prefix, n_pairs=len(pairs), n_matched=coverage["matched"],
                              n_missing=coverage["missing_condition"], n_not_admitted=coverage["not_admitted"],
                              n_a_only_wins=deltas.count(-1), n_b_only_wins=deltas.count(1),
                              mean_success_delta_b_minus_a=ratio(sum(deltas), len(deltas))))
    return summaries, pair_rows


def failures(episodes):
    for row in episodes:
        failure = row["failure"]
        if failure is not None:
            yield {**{field: row[field] for field in
                      ("run_lock_sha256", "task", "episode_id", "seed", "comparison_id", "pair_id",
                       "match_status", "outcome", "episode_artifact")},
                   "condition_id": row["condition"]["id"],
                   "action_source": row["condition"]["action_source"], **failure}


def write_csv(path, rows, data_kind, fields=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["data_kind", *(fields or rows[0])])
        writer.writeheader()
        writer.writerows(dict(row, data_kind=data_kind) for row in rows)


OUTCOMES = ("success", "initial_state_mismatch", "environment_error", "interrupted", "ik_unreachable",
            "ik_failed", "planning_failed", "joint_limit", "adapter_error", "control_error", "policy_failed",
            "task_unsuccessful")
FAILURE_FIELDS = ("run_lock_sha256", "task", "episode_id", "seed", "comparison_id", "pair_id", "match_status",
                  "outcome", "episode_artifact", "condition_id", "action_source", "stage", "code", "detail",
                  "step", "artifact")
PAIR_FIELDS = ("run_lock_sha256", "task", "comparison_id", "condition_a", "condition_b", "pair_id", "status",
               "success_delta_b_minus_a", "episode_a", "episode_b")
CONTRAST_FIELDS = (*PAIR_FIELDS[:5], "n_pairs", "n_matched", "n_missing", "n_not_admitted",
                   "n_a_only_wins", "n_b_only_wins", "mean_success_delta_b_minus_a")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--data-kind", choices=("synthetic", "diagnostic", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--run-lock", type=Path, help="frozen roster JSON, required for formal inputs")
    args = parser.parse_args()
    if (args.data_kind == "formal") != (args.run_lock is not None):
        parser.error("--run-lock is required for formal inputs and only accepted in formal mode")
    try:
        episodes = load_records(args.inputs, args.data_kind)
        if args.run_lock is not None:
            check_formal_roster(args.run_lock, episodes)
        conditions = list(condition_tables(episodes))
        contrasts, pairs = paired_tables(episodes)
        if args.data_kind == "formal" and any(row["n_missing"] for row in contrasts):
            raise ValueError("formal contrasts require both conditions in every locked pair")
        args.output.mkdir(parents=True, exist_ok=False)
        write_csv(args.output / "condition_summary.csv", conditions, args.data_kind)
        write_csv(args.output / "paired_summary.csv", contrasts, args.data_kind, CONTRAST_FIELDS)
        write_csv(args.output / "pair_index.csv", pairs, args.data_kind, PAIR_FIELDS)
        write_csv(args.output / "failure_index.csv", list(failures(episodes)), args.data_kind, FAILURE_FIELDS)
        (args.output / "episodes.json").write_text(
            json.dumps(episodes, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"{args.data_kind}: {len(episodes)} episodes, {len(conditions)} conditions, {len(pairs)} pair IDs")


if __name__ == "__main__":
    main()
