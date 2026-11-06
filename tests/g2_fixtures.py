"""Synthetic alignment fixture using real, SHA-checked Aloha/ARX URDFs."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g2_adapter import Adapter, ArmSample, Snapshot, Stamp, CAMERAS
from g2_assets import load_arms
from g2_motion import QuinticMotion
from robotwin_control_geometry import inverse_pose, pose_from_wxyz

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "research/robotwin/embodiment_config_audit.json"
ASSETS = ROOT / "research/robotwin/embodiment_assets"


def robot_arms(robot):
    calibration = {}
    for side, prefix in zip(("left", "right"), ("fl_", "fr_") if robot == "aloha-agilex" else ("", "")):
        calibration[side] = dict(
            finger_names=[prefix + "joint7", prefix + "joint8"],
            closed_m=[0., 0.], opened_m=[.04, .04], grip_source="synthetic_fixture_only",
            instance=("aloha" if robot == "aloha-agilex" else "arx_" + side),
            child_joint=np.eye(4).tolist(), tcp_tool=np.eye(4).tolist(), id="synthetic_nominal_tcp_v1")
    return load_arms(ASSETS, AUDIT, robot, calibration)


def fixture():
    aloha, arx = robot_arms("aloha-agilex"), robot_arms("ARX-X5")
    qa = np.array([[.2, .3, -.4, .3, .4, -.2], [-.2, .4, -.3, .2, -.3, .2]])
    qr = np.array([[.3, .2, -.3, .4, .3, -.1], [-.1, .3, -.2, .3, -.4, .2]])
    source_bases = [pose_from_wxyz([i * .5, -.65, 0], [.707, 0, 0, .707]) for i in (-1, 1)]
    # Synthetic base registration makes one pose exactly common. It is never a scene calibration.
    target_bases = [base @ a.chain.fk(q_a) @ a.offset() @ inverse_pose(r.chain.fk(q_r) @ r.offset())
                    for base, a, r, q_a, q_r in zip(source_bases, aloha, arx, qa, qr)]
    motion = QuinticMotion(.02, np.full((2, 6), .5), np.ones((2, 6)),
                           np.full((2, 2), .03), np.full((2, 2), .1),
                           .02, .001, "synthetic_limits_only", "mock_free_space_v1")
    adapter = Adapter(aloha, arx, source_bases, np.eye(4), qa,
                      np.ones((2, 6)), lambda q, g: True, lambda q, g: True, motion,
                      grip_rule="clip", grip_max_clip=.05)
    return adapter, qa, qr, target_bases


def snapshot(adapter, q, bases, tick=0, episode="mock", grip=.5):
    stamp = Stamp(episode, tick, tick * .1)
    samples = []
    for a, values, base in zip(adapter.arx, q, bases):
        names = a.chain.names + a.grip.names
        actual = np.r_[values, a.grip.target(grip)]
        # Reverse order to ensure selection is by name rather than index.
        samples.append(ArmSample(stamp, a.instance, names[::-1], actual[::-1], base.copy()))
    return Snapshot(stamp, tuple(samples),
                    {name: (stamp, np.full((3, 16, 16), tick, dtype=np.uint8)) for name in CAMERAS}, np.eye(4))
