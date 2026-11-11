"""Read-only analysis of historical and projection-enabled G2 ARX results."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


REJECTION_EVENTS = {
    "original_action_rejection", "original_proxy_rejection", "source_observability_barrier",
    "task_tcp_projection_rejected", "planning_gate_rejection", "contact_failure",
}
RGB_CAMERAS = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
CONTACT_STAGES = {"left_holding": "grasp", "both_holding": "shared_hold",
                  "right_holding": "right_hold", "free_after_hold": "released"}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def jsonl(path):
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError as error:
                # A process killed at its deadline can leave a partial final line.
                if not line.endswith(b"\n"):
                    yield number, None
                    return
                raise ValueError(f"{path.name}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number}: expected an object")
            yield number, row


def collision_pairs(value):
    pairs = set()
    if isinstance(value, dict):
        if value.get("link_name1") and value.get("link_name2"):
            pairs.add(" / ".join(sorted((value["link_name1"], value["link_name2"]))))
        for child in value.values():
            pairs.update(collision_pairs(child))
    elif isinstance(value, list):
        for child in value:
            pairs.update(collision_pairs(child))
    return sorted(pairs)


def audit_world(path):
    checks = blocked = 0
    pairs = Counter()
    partial = False
    for _, row in jsonl(path):
        if row is None:
            partial = True
            break
        checks += 1
        if row["blocked"]:
            blocked += 1
            pairs.update(collision_pairs(row["blocked"]))
    return dict(checks=checks, blocked_checks=blocked, link_pairs=pairs,
                partial_final_line=partial,
                scope="FCL audit calls, including planning and/or measured checks")


def analyze_events(rows):
    """Count measured samples and link requests to the projection that preceded them."""
    queries = []
    event_counts = Counter()
    phases = Counter()
    contact_ticks = Counter()
    holders = Counter()
    stage_evidence = {}
    projections = {}
    pending_reobservation = None
    rejections = []
    reobserve_rejections = []
    physical_steps = 0
    last_measured = None
    partial_line = None

    def marker(row, line):
        return dict(tick=row.get("tick"), sim_s=row.get("sim_s"),
                    query=len(queries) or None, line=line)

    for line, row in rows:
        if row is None:
            partial_line = line
            break
        kind = row.get("kind")
        if kind is None:
            if not row.get("physics_step_count"):
                continue
            kind = "executed_step"
        event_counts[kind] += 1
        current = queries[-1] if queries else None
        if kind == "policy_request":
            stamp = row["stamp"]
            source = row.get("after_projection_from")
            query = dict(query=len(queries) + 1, stamp=stamp, response_logged=False,
                         response_validated=False, approved_plan=False, physical_steps=0,
                         completed_actions_logged=None, after_projection_from=source)
            if source is not None:
                projection = projections.get((source["episode"], source["tick"]))
                advanced = (stamp["episode"] == source["episode"] and stamp["tick"] > source["tick"]
                            and stamp["sim_s"] > source["sim_s"])
                query["projection_reobservation"] = dict(
                    linked=projection is not None, clock_advanced=advanced,
                    arms_logged=len(row.get("actual") or []) == 2,
                    objects_logged=sorted((row.get("objects_task") or {}).keys()),
                )
                if projection is not None and advanced:
                    projection["fresh_request"] = query["query"]
            pending_reobservation = None
            queries.append(query)
        elif kind in ("policy_response", "policy_validated", "approved_plan") and current is not None:
            field = {"policy_response": "response_logged", "policy_validated": "response_validated",
                     "approved_plan": "approved_plan"}[kind]
            current[field] = True
        elif kind == "task_tcp_projection":
            stamp = row["stamp"]
            projections[(stamp["episode"], stamp["tick"])] = dict(
                query=len(queries), stamp=stamp, method=row.get("method"),
                translation_m=row.get("left_task_translation_m"),
                candidates=row.get("attempts_count", len(row.get("attempts", ()))),
                executed_actions=None, discarded_rows=None, fresh_request=None,
            )
        elif kind in ("projection_chunk_discarded", "measured_replan"):
            if current is not None:
                current["completed_actions_logged"] = row.get("executed_actions")
            if kind == "projection_chunk_discarded":
                stamp = row["stamp"]
                projection = projections.get((stamp["episode"], stamp["tick"]))
                if projection is not None:
                    projection.update(executed_actions=row.get("executed_actions"),
                                      discarded_rows=row.get("discarded_rows"))
                    pending_reobservation = stamp
        if kind in REJECTION_EVENTS:
            diag = row.get("source_diagnostic") or row.get("diagnostic") or {}
            scope = "source_collision" if kind in ("source_observability_barrier", "original_proxy_rejection") else diag.get("collision_world")
            rejection = dict(kind=kind, **marker(row, line), status=row.get("source_status", row.get("status")),
                             geometry=row.get("geometry") or diag.get("geometry"),
                             world=scope, link_pairs=collision_pairs(row),
                             continuity_rejected=diag.get("continuity_rejected"),
                             collision_candidates_rejected=diag.get("rejected_pairs"))
            rejections.append(rejection)
            if kind == "original_proxy_rejection":
                # Observation occurs before the next policy_request event.
                reobserve_rejections.append(dict(rejection, next_query_number=len(queries) + 1,
                                                 after_projection_from=pending_reobservation))
        steps = row.get("physics_step_count") or 0
        physical_steps += steps
        if steps:
            last_measured = marker(row, line)
        if current is not None:
            current["physical_steps"] += steps
        contact = row.get("contact_state")
        if not steps or not isinstance(contact, dict):
            continue
        phase = contact.get("phase")
        if phase:
            phases[phase] += steps
        stage = CONTACT_STAGES.get(phase)
        if stage:
            stage_evidence.setdefault(stage, marker(row, line))
        if phase == "right_holding" and (
            contact.get("saw_shared_hold") or "shared_hold" in stage_evidence
        ):
            stage_evidence.setdefault("handover", marker(row, line))
        for side, fingers in (contact.get("contact_fingers") or {}).items():
            if fingers:
                contact_ticks[side] += steps
                stage_evidence.setdefault(f"{side}_contact", marker(row, line))
        for side in contact.get("holders") or ():
            holders[side] += steps

    return dict(queries=queries, counts=event_counts, physical_steps=physical_steps,
                contact_phase_ticks=phases, finger_contact_ticks=contact_ticks, holder_ticks=holders,
                stages=stage_evidence, projection_plans=list(projections.values()),
                rejection_events=rejections, observation_rejections=reobserve_rejections,
                last_measured=last_measured, partial_final_line=partial_line)


def analyze(run, episode=None, collision_dir=None, expected=None):
    hashes = {}
    checks = []

    def artifact(relative, path=None):
        path = run / relative if path is None else path
        if not path.is_file():
            return None
        key = str(relative)
        hashes[key] = digest(path)
        if expected and key in expected:
            if hashes[key] != expected[key]:
                raise ValueError(f"archived input hash differs: {key}")
            checks.append(key)
        return path

    def document(relative):
        path = artifact(relative)
        return read(path) if path else {}

    standalone = document(Path("standalone-result.json"))
    authorization = document(Path("authorization-receipt.json"))
    campaign = document(Path("supervisor/campaign.json"))
    deployment_error = document(Path("deployment-failure.json"))
    episode = episode or (standalone.get("selected") or {}).get("episode") or authorization.get("episode")
    if not episode:
        trials = sorted(p.name for p in (run / "trials").glob("*") if p.is_dir())
        if len(trials) != 1:
            raise ValueError("specify --episode for a multi-trial or incomplete archive")
        episode = trials[0]
    if Path(episode).name != episode or episode in ("", ".", ".."):
        raise ValueError("invalid episode name")
    trial = Path("trials") / episode
    folder = trial / "episode"
    startup = document(Path("startup") / episode / "receipt.json")
    receipt = document(trial / "receipt.json")
    row = (campaign.get("rows") or {}).get(episode) or standalone.get("row") or {}
    if standalone.get("row") and row != standalone["row"]:
        raise ValueError("supervisor and standalone row disagree")
    if receipt and row.get("result") and receipt != row["result"]:
        raise ValueError("worker and supervisor receipt disagree")
    receipt = receipt or row.get("result") or {}
    worker = document(folder / "result.json")
    final_worker = receipt.get("result") or {}
    if final_worker and worker and final_worker != worker:
        raise ValueError("worker result and final receipt disagree")
    result = final_worker or worker
    events = artifact(folder / "events.jsonl")
    activity = analyze_events(jsonl(events)) if events else None
    failures = result.get("failure") or {}
    diagnostic = failures.get("diagnostic") or {}
    status = standalone.get("outcome") or result.get("outcome") or receipt.get("outcome")
    if not status or status == "running":
        status = deployment_error.get("outcome") or status
    row_reason = row.get("reason")
    if row.get("status") == "not_run" and row_reason == "not reached":
        row_reason = None
    stop = row_reason or standalone.get("stop_reason") or campaign.get("stop_reason")
    if stop == "trial hard timeout" or status == "timeout" or failures.get("code") == "timeout":
        status = "RESOURCE_CENSORED"
    finalized = bool(standalone or receipt or stop or status not in (None, "running"))
    recorded_success = receipt.get("success") if "success" in receipt else result.get("success")
    success = recorded_success if finalized and status != "RESOURCE_CENSORED" else None

    # At a host kill, result.json can be the last checkpoint while events
    # contain a later, interrupted query. Keep both counters and the discrepancy.
    query_count = activity["counts"].get("policy_request", 0) if activity else None
    if activity and result.get("queries") is not None and query_count != result["queries"]:
        if receipt.get("result") and activity["partial_final_line"] is None:
            raise ValueError("final worker query count differs from complete events")
    for query in activity["queries"] if activity else ():
        number = query["query"]
        action = artifact(folder / f"action-{number:05d}.npy")
        query["action_file"] = action.name if action else None
        if action:
            try:
                chunk = np.load(action, allow_pickle=False)
                valid = chunk.shape == (50, 14) and bool(np.isfinite(chunk).all())
            except (ValueError, EOFError, OSError, TypeError):
                valid = False
            query["saved_action_valid"] = valid
            if valid:
                query["first_gripper_row_at_or_below_0_5"] = [
                    int(rows[0] + 1) if len(rows) else None
                    for rows in (np.flatnonzero(chunk[:, 6] <= .5), np.flatnonzero(chunk[:, 13] <= .5))
                ]
        observation = artifact(folder / f"observation-{number:05d}.npz")
        query["observation_file"] = observation.name if observation else None
        reobservation = query.get("projection_reobservation")
        if reobservation is not None and observation:
            try:
                with np.load(observation, allow_pickle=False) as rgb:
                    reobservation["rgb_cameras"] = sorted(rgb.files)
                    reobservation["saved_rgb_valid"] = set(rgb.files) == RGB_CAMERAS and all(
                        image.dtype == np.uint8 and image.ndim == 3 and image.shape[0] == 3
                        and all(0 < side <= 1024 for side in image.shape[1:])
                        for image in (rgb[name] for name in RGB_CAMERAS)
                    )
            except (ValueError, EOFError, OSError, TypeError):
                reobservation["saved_rgb_valid"] = False
    worlds = {}
    for key, leaf in (("target", "collision-checks.jsonl"), ("source_proxy", "proxy-collision-checks.jsonl")):
        path = artifact(folder / leaf, collision_dir / leaf if collision_dir else None)
        worlds[key] = audit_world(path) if path else None
    terminal_pairs = collision_pairs(diagnostic)
    collision_world = diagnostic.get("collision_world")
    collision_world_evidence = "explicit diagnostic" if collision_world else None
    if not collision_world and terminal_pairs and diagnostic.get("collision_examples"):
        example_hits = [hit for example in diagnostic["collision_examples"] for hit in example.get("collisions", ())]
        if example_hits and all(str(hit.get(f"object_name{i}", "")).startswith("g2_aloha_proxy")
                                for hit in example_hits for i in (1, 2)):
            collision_world = "source_collision"
            collision_world_evidence = "terminal collision object names"
    budget_stop = None
    if failures.get("code") == "action_budget_exhausted" and result.get("queries") is not None:
        budget_stop = "query_limit_reached" if result["queries"] >= 300 else "policy_action_limit_reached"
    return dict(job_id=standalone.get("job_id") or authorization.get("job_id"), episode=episode,
                action_source="expert" if "expert" in episode else "policy",
                counters=dict(policy_actions=result.get("policy_actions"),
                              executed=result.get("executed"), recovery_actions=result.get("recovery_actions"),
                              requested_prefix_rows=result.get("requested"), result_queries=result.get("queries"),
                              logged_model_requests=query_count,
                              evidence="worker receipt" if final_worker else "worker result checkpoint" if worker else None),
                activity=activity, collision_audits=worlds,
                startup={key: startup[key] for key in ("category", "physical_admitted", "fresh_reset_verified")
                         if key in startup},
                termination=dict(finalized=finalized, task_success=success, outcome=status,
                                 worker_outcome=result.get("outcome"), category=receipt.get("category"),
                                 supervisor_status=row.get("status"), supervisor_reason=stop,
                                 failure_phase=failures.get("phase") or deployment_error.get("stage"),
                                 failure_code=failures.get("code") or deployment_error.get("error"),
                                 budget_stop=budget_stop,
                                 failure_detail=failures.get("detail") or deployment_error.get("detail"),
                                 geometry=diagnostic.get("geometry"), collision_world=collision_world,
                                 collision_world_evidence=collision_world_evidence, terminal_link_pairs=terminal_pairs,
                                 continuity_rejected=diagnostic.get("continuity_rejected"),
                                 collision_candidates_rejected=diagnostic.get("rejected_pairs"),
                                 projection_stop_reason=diagnostic.get("projection_stop_reason"),
                                 cleanup=receipt.get("cleanup"), process_group_reaped=row.get("process_group_reaped")),
                placement=dict(native_success_predicate=success,
                               last_measured_tick=(activity.get("last_measured") or {}).get("tick")
                               if success is True and activity else None),
                source_sha256=hashes, verified_source_members=sorted(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode")
    parser.add_argument("--collision-dir", type=Path)
    parser.add_argument("--expected-hashes", type=Path, help="publication JSON containing source_sha256")
    args = parser.parse_args()
    expected = read(args.expected_hashes)["source_sha256"] if args.expected_hashes else None
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error("output must be outside the raw run directory")
    report = analyze(args.run, args.episode, args.collision_dir, expected)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
