"""Stdlib-only verification of author H50 receipts on the allocation host."""

from hashlib import sha256
import json
import re

from g2_author_h50_limits import MAX_ACTIONS, MODE, WINDOW
from g2_candidate_stop import (
    actual_provenance,
    candidate_failure,
    verified_candidate_stop,
)
from g2_final_campaign import verified_scene_cleanup


def clock_sha256(clock):
    return sha256(
        json.dumps(
            clock, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def measured_stop_proof(proof, receipt):
    return (
        isinstance(proof, dict)
        and proof.get("episode") == receipt["episode_id"]
        and type(proof.get("tick")) is int
        and proof["tick"] >= 0
        and all(
            proof.get(k) is True
            for k in (
                "terminal_checks_passed",
                "hold_verified",
                "hold_complete",
                "close_complete",
                "scene_context_closed",
                "no_physics_advance",
                "contact_sample_measured",
            )
        )
        and proof.get("contact_source") == "sapien.Scene.get_contacts"
        and type(proof.get("contact_manifold_count")) is int
        and proof["contact_manifold_count"] >= 0
        and all(
            isinstance(proof.get(k), str) and re.fullmatch("[a-f0-9]{64}", proof[k])
            for k in ("state_sha256", "contact_sha256")
        )
        and (
            proof["tick"] != 0
            or (
                receipt.get("initial_measurement", {}).get("status") == "measured"
                and all(
                    proof[k] == receipt["initial_measurement"].get(k)
                    for k in ("state_sha256", "contact_sha256")
                )
            )
        )
    )


def verified_resource_stop(receipt):
    proof = receipt.get("resource_stop")
    clock = receipt.get("trial_clock") or {}
    reason = receipt.get("resource_reason")
    return (
        receipt.get("outcome") == "RESOURCE_CENSORED"
        and receipt.get("success") is False
        and receipt.get("request_in_flight") is False
        and measured_stop_proof(proof, receipt)
        and proof.get("kind") == "resource_stop"
        and proof.get("resource_reason") == reason
        and proof.get("trial_clock_sha256") == clock_sha256(clock)
        and proof["tick"] == receipt.get("physics_steps")
        and (
            reason == "physics_steps"
            and proof["tick"] == (clock.get("budget") or {}).get("physics_steps")
            or reason == "soft_wall_clock"
            and type(clock.get("soft")) in (float, int)
            and type(receipt.get("finished_at")) in (float, int)
            and clock["soft"] <= receipt["finished_at"]
        )
    )


def may_continue(
    receipt, expected_episode, *, process_group_reaped, shared_service_alive, clock=None
):
    """Require complete scene proof, isolation, and the host's original clock."""
    if (
        process_group_reaped is not True
        or shared_service_alive is not True
        or receipt.get("mode") != MODE
        or receipt.get("episode_id") != expected_episode
        or receipt.get("stop_campaign") is True
        or actual_provenance(
            ((receipt.get("result") or {}).get("failure") or {}).get("diagnostic")
        )
        or not verified_scene_cleanup(receipt)
    ):
        return False
    if clock is not None:
        finished = receipt.get("finished_at")
        completed, native = (
            receipt.get("completed_actions"),
            receipt.get("native_action_count"),
        )
        ticks, queries = (
            receipt.get("physics_steps"),
            receipt.get("query_action_counts"),
        )
        if (
            receipt.get("trial_clock") != clock.receipt()
            or type(finished) not in (float, int)
            or not clock.started <= finished < clock.reap_at
            or type(completed) is not int
            or type(native) is not int
            or not 0 <= completed <= native <= min(completed + 1, MAX_ACTIONS)
            or type(ticks) is not int
            or not 0 <= ticks <= clock.budget.physics_steps
            or not isinstance(queries, list)
            or queries != list(range(0, len(queries) * WINDOW, WINDOW))
            or queries
            and not queries[-1] <= completed
        ):
            return False
    if receipt.get("category") == "diagnostic_complete":
        if receipt.get("outcome") == "SUCCESS":
            return receipt["success"] is True and (
                clock is None or 0 < completed == native
            )
        return (
            receipt.get("outcome") == "TASK_INCOMPLETE"
            and receipt["success"] is False
            and (
                clock is None
                or completed == native == MAX_ACTIONS
                and len(queries) == MAX_ACTIONS // WINDOW
            )
        )
    if receipt.get("category") == "resource_censored":
        return clock is not None and verified_resource_stop(receipt)
    proof = receipt.get("candidate_stop")
    return (
        receipt.get("category") == "controlled_candidate_stop"
        and receipt["success"] is False
        and candidate_failure(receipt.get("result") or {})
        and verified_candidate_stop(proof, expected_episode, receipt["result"])
        and measured_stop_proof(proof, receipt)
        and (clock is None or proof["tick"] == receipt.get("physics_steps"))
    )
