"""Concrete native/ARX conditions for the single proposed campaign.

No Slurm, GPU initialization or model loading. Scene factories are supplied by
an eventual approved frozen job. The currently running release never imports it.
"""

from copy import deepcopy
from dataclasses import asdict
import numpy as np
from g2_remaining_episode import run as run_recovery
from g2_remaining_router import JointRouter
from g2_episode import run_episode, save_json
from g2_export import episode_record
from g2e_protocol import PROTOCOL, CONTACT_MODE_NATIVE
from g2_task_runtime import validate_config


def configure(base, row, *, allow_native_h50=False):
    config = deepcopy(base)
    if row["embodiment"] != base["embodiment"] or row["seed"] != base["scene_seed"]:
        raise ValueError("condition reset/embodiment mismatch")
    if "expert" in row["method"]:
        raise ValueError(
            "expert control requires the explicit official replay and placement stage entry"
        )
    config.update(
        episode_id=row["episode_id"],
        condition=row["condition"],
        window=row["window"],
        max_actions=row["max_policy_actions"],
        timeout_s=row["hard_timeout_s"] - 50,
        policy_rng_seed=row["policy_rng_seed"],
        formal_release_ready=False,
    )
    if row["embodiment"] == "aloha-agilex":
        config.update(
            execution_protocol=PROTOCOL,
            contact_mode=CONTACT_MODE_NATIVE,
            native_release_semantics="physical_aperture_v1",
        )
    elif "execution_protocol" in config:
        raise ValueError("native protocol cannot leak into ARX")
    validate_config(config, allow_native_h50=allow_native_h50)
    return config


def run_condition(config, row, factory, policy, output, *, allow_native_h50=False):
    """Injected factory must own cleanup if construction/reset fails."""
    from pathlib import Path

    output = Path(output)
    projection = config.get("task_tcp_projection")
    if projection is not None and (row["embodiment"] != "ARX-X5"
            or row["method"] not in ("bounded_recovery", "bounded_recovery_and_router")
            or not row["recovery_cap"]):
        raise ValueError("TCP projection requires an ARX policy recovery trial")
    spec = dict(
        data_kind="diagnostic",
        episode_id=config["episode_id"],
        condition=config["condition"],
        prompt=config["prompt"],
        window=config["window"],
        max_actions=config["max_actions"],
        timeout_s=config["timeout_s"],
        max_recoveries=row["recovery_cap"],
        native_grip_max_clip=config["native_grip_max_clip"],
        approach_repair=config.get("free_space_approach_repair") is not None,
        task_tcp_projection=projection,
    )
    if row["embodiment"] == "aloha-agilex":
        from g2_run_task import execute

        return execute(
            config, factory, policy, output, allow_native_h50=allow_native_h50
        )
    # Defer scene creation until the harness has created its evidence directory,
    # just like RuntimeSession; this also permits normal reset/cleanup on faults.
    from g2_task_runtime import RuntimeSession, SessionAdapter

    session = RuntimeSession(factory)
    session._drive_fingers = lambda: session.backend._drive_fingers()
    original = session.reset

    def reset(spec):
        actual = original(spec)
        if row["method"] == "bounded_recovery_and_router":
            session.adapter.motion = JointRouter(
                session.adapter.motion,
                np.array([a.chain.limits for a in session.adapter.arx]),
            )
        return actual

    session.reset = reset
    if row["method"] == "frozen_exact_mapping":
        return run_episode(
            spec, session, policy, output, adapter=SessionAdapter(session)
        )
    if row["method"] not in ("bounded_recovery", "bounded_recovery_and_router"):
        raise ValueError("unknown recovery condition")

    # Adapter is created inside reset. Proxy forwards only adapter methods/state;
    # it never maps or fabricates observations itself.
    class AdapterProxy:
        def __getattr__(self, name):
            return getattr(session.adapter, name)

    return run_recovery(spec, session, AdapterProxy(), policy, output)
