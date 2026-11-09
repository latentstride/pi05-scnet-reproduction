"""CPU-only exact-pose IK refinement and explicit bounded progress diagnostics.

This module is opt-in. It does not change any running/frozen release. A safe
partial goal is reported as a changed command, never as reaching the policy goal.
"""

from dataclasses import replace
import numpy as np
from g2_adapter import G2Error, solve_pair
from g2_kinematics import rotation_vector, pose_error
from verify_robotwin_fk import rot
from robotwin_control_geometry import checked_pose, finite_array, Status
from g2_candidate_stop import actual_provenance


class RefinedChain:
    def __init__(self, chain, position, angle):
        self.chain, self.position, self.angle = chain, position, angle

    def __getattr__(self, name):
        return getattr(self.chain, name)

    def ik(self, goal, seed, check, **kwargs):
        return self.chain.ik(
            goal,
            seed,
            check,
            pos_tol=min(kwargs.get("pos_tol", 1.0), self.position),
            ang_tol=min(kwargs.get("ang_tol", 1.0), self.angle),
        )


def refined_pair(arms, frames, tools, seed, fingers, world, check, max_delta=None):
    verdicts = []
    for position, angle in ((0.00025, np.deg2rad(0.05)), (1e-5, 1e-5), (1e-7, 1e-6)):
        check()
        refined = tuple(
            replace(a, chain=RefinedChain(a.chain, position, angle)) for a in arms
        )
        try:
            q, residual = solve_pair(
                refined, frames, tools, seed, fingers, world, check, max_delta
            )
            return (
                q,
                residual,
                dict(
                    refinement_position_m=position,
                    refinement_angle_rad=angle,
                    attempts=verdicts,
                ),
            )
        except G2Error as exc:
            if actual_provenance(exc.diagnostic) or exc.status not in (
                Status.COLLISION,
                Status.IK_FAILED,
                Status.VELOCITY,
            ):
                raise
            verdicts.append(
                dict(
                    position_m=position,
                    angle_rad=angle,
                    status=exc.status.value,
                    detail=str(exc),
                    diagnostic=exc.diagnostic,
                )
            )
    raise G2Error(
        Status.COLLISION,
        "bounded exact-pose refinements exhausted",
        geometry="candidate",
        reason="exact_pose_refinements_exhausted",
        attempts=verdicts,
    )


def interpolate_tools(start, end, fraction):
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("invalid tool progress")
    result = []
    for a, b in zip(finite_array(start, (2, 4, 4)), finite_array(end, (2, 4, 4))):
        a, b = checked_pose(a), checked_pose(b)
        axis = rotation_vector(b[:3, :3] @ a[:3, :3].T)
        angle = float(np.linalg.norm(axis))
        p = a.copy()
        p[:3, 3] = (1 - fraction) * a[:3, 3] + fraction * b[:3, 3]
        p[:3, :3] = (
            rot(axis / angle, angle * fraction) if angle > 1e-12 else np.eye(3)
        ) @ a[:3, :3]
        result.append(checked_pose(p))
    return np.asarray(result)


def bounded_progress(arms, frames, tools, q, fingers, requested, world, motion, check):
    """Free-space approach only. Holding/release must use contact-specific paths.

    The minimum retained fraction is 1/16. No collision exemptions or relaxed IK
    residuals. Geometry and complete retimed paths must both pass. Static CPU
    acceptance is not authorization to execute a new policy action.
    """
    if getattr(world, "tracker", None) is not None and (
        world.tracker.holders or world.tracker.releasing
    ):
        raise G2Error(Status.INVALID, "partial-goal recovery forbidden while holding")
    start = np.array(
        [f @ a.chain.fk(v) @ a.offset() for a, f, v in zip(arms, frames, q)]
    )
    attempts = []
    for fraction in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625):
        check()
        candidate = interpolate_tools(start, tools, fraction)
        deviations = [pose_error(a, b) for a, b in zip(candidate, tools)]
        if any(
            np.linalg.norm(e[:3]) > 0.012 or np.rad2deg(np.linalg.norm(e[3:])) > 1.0
            for e in deviations
        ):
            attempts.append(
                dict(
                    fraction=fraction,
                    status="command_change_bound",
                    detail="changed endpoint exceeds 12mm/1deg envelope",
                )
            )
            continue
        try:
            goal, residual, refinement = refined_pair(
                arms, frames, candidate, q, requested, world, check
            )
            path = motion.plan(q, fingers, [(goal, requested)], world, check)
            return dict(
                goal_q=goal,
                path=path,
                executed_tools=candidate,
                original_tools=np.asarray(tools).copy(),
                fraction=fraction,
                original_goal_reached=fraction == 1.0,
                pose_change_m=[
                    float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
                    for a, b in zip(candidate, tools)
                ],
                refinement=refinement,
                attempts=attempts,
            )
        except G2Error as exc:
            if actual_provenance(exc.diagnostic) or exc.status not in (
                Status.COLLISION,
                Status.IK_FAILED,
                Status.VELOCITY,
                Status.PLAN_FAILED,
            ):
                raise
            attempts.append(
                dict(
                    fraction=fraction,
                    status=exc.status.value,
                    detail=str(exc),
                    diagnostic=exc.diagnostic,
                )
            )
    raise G2Error(
        Status.COLLISION,
        "no safe bounded progress endpoint and full path",
        geometry="candidate",
        reason="bounded_progress_exhausted",
        attempts=attempts,
    )
