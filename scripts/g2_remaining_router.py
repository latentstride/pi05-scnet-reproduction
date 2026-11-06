"""Bounded, deterministic free-space joint rerouting under the existing checker.

Goal q, gripper intent, original action accounting, limits and the final dense
retimer remain unchanged. Held-object/contact motion is deliberately not sampled
as unconstrained free space; it needs the existing phase-specific planner.
"""

from dataclasses import replace
from time import monotonic
import numpy as np
from g2_adapter import G2Error, clear
from robotwin_control_geometry import Status, finite_array


class JointRouter:
    method = "g2_bounded_free_space_rrt_v1"

    def __init__(self, motion, limits, seed=20261106, max_nodes=384):
        self.motion = motion
        self.limits = finite_array(limits, (2, 6, 2))
        if (
            np.any(self.limits[:, :, 0] >= self.limits[:, :, 1])
            or type(max_nodes) is not int
            or max_nodes < 2
        ):
            raise ValueError("invalid routing bounds")
        self.seed, self.max_nodes = seed, max_nodes
        self.last = {}

    def __getattr__(self, name):
        return getattr(self.motion, name)

    def route(self, start, goal, fingers, collision, check):
        start = finite_array(start, (2, 6))
        goal = finite_array(goal, (2, 6))
        fingers = finite_array(fingers, (2, 2))
        for q in (start, goal):
            if np.any(q < self.limits[:, :, 0]) or np.any(q > self.limits[:, :, 1]):
                raise G2Error(Status.LIMIT, "route endpoint outside joint limits")
            if not clear(collision, q, fingers):
                raise G2Error(
                    Status.COLLISION, "route endpoint collision cannot be bypassed"
                )
        nodes = [start.copy()]
        parents = [None]
        rng = np.random.default_rng(self.seed)
        edge_calls = 0

        def edge(a, b):
            nonlocal edge_calls
            count = max(
                1,
                int(np.ceil(np.max(abs(b - a)) / float(min(self.motion.sampling[:6])))),
            )
            for t in np.linspace(0, 1, count + 1):
                check()
                edge_calls += 1
                if not clear(collision, a + (b - a) * t, fingers):
                    return False
            return True

        for iteration in range(self.max_nodes - 1):
            check()
            if iteration % 4 == 0:
                sample = goal
            elif iteration % 4 == 1:
                sample = start + rng.normal(0, 0.6, (2, 6))
            else:
                sample = rng.uniform(self.limits[:, :, 0], self.limits[:, :, 1])
            nearest = min(
                range(len(nodes)), key=lambda i: np.linalg.norm(nodes[i] - sample)
            )
            delta = sample - nodes[nearest]
            candidate = nodes[nearest] + delta * min(
                1.0, 0.25 / max(float(np.linalg.norm(delta)), 1e-12)
            )
            candidate = np.clip(candidate, self.limits[:, :, 0], self.limits[:, :, 1])
            if not edge(nodes[nearest], candidate):
                continue
            nodes.append(candidate)
            parents.append(nearest)
            if edge(candidate, goal):
                path = [goal.copy()]
                index = len(nodes) - 1
                while index is not None:
                    path.append(nodes[index])
                    index = parents[index]
                path = list(reversed(path))
                # Deterministic farthest-visible shortcut, all edges rechecked.
                result = [path[0]]
                i = 0
                while i < len(path) - 1:
                    j = next(
                        j for j in range(len(path) - 1, i, -1) if edge(path[i], path[j])
                    )
                    result.append(path[j])
                    i = j
                self.last = dict(
                    nodes=len(nodes),
                    iterations=iteration + 1,
                    edge_samples=edge_calls,
                    seed=self.seed,
                )
                return result
        self.last = dict(
            nodes=len(nodes),
            iterations=self.max_nodes - 1,
            edge_samples=edge_calls,
            seed=self.seed,
        )
        raise G2Error(
            Status.PLAN_FAILED, "bounded joint routing exhausted", **self.last
        )

    def plan(self, q, fingers, targets, collision, check):
        targets = list(targets)
        started = monotonic()
        try:
            path = self.motion.plan(q, fingers, targets, collision, check)
            self.last = dict(rerouted=False, wall_s=monotonic() - started)
            return path
        except G2Error as exc:
            if exc.status != Status.COLLISION:
                raise
        tracker = getattr(collision, "tracker", None)
        if tracker is not None and (tracker.holders or tracker.releasing):
            raise G2Error(
                Status.COLLISION,
                "free-space rerouting forbidden during contact/holding",
            )
        drive = (
            collision.drive_start(fingers)
            if callable(getattr(collision, "drive_start", None))
            else fingers
        )
        if any(not np.allclose(g, drive, atol=1e-12, rtol=0) for _, g in targets):
            raise G2Error(
                Status.COLLISION,
                "rerouting requires a fixed gripper; split contact phases explicitly",
            )
        routed = []
        indices = []
        start = np.asarray(q)
        for goal, g in targets:
            route = self.route(start, goal, np.asarray(fingers), collision, check)
            routed.extend((point, g) for point in route[1:])
            indices.append(len(routed) - 1)
            start = goal
        path = self.motion.plan(q, fingers, routed, collision, check)
        # Internal RRT vertices are NOT extra policy actions.
        events = tuple(path.waypoint_ticks[i] for i in indices)
        self.last.update(
            rerouted=True,
            policy_waypoints=len(targets),
            internal_waypoints=len(routed),
            wall_s=monotonic() - started,
        )
        return replace(path, waypoint_ticks=events)
