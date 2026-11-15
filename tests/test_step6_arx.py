import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.step6_arx.analyze import summarize
from scripts.step6_arx.metrics import Stages
from scripts.step6_arx.protocol import SHAPE, actions_from, request_from
from scripts.step5_arx.data import CAMERAS


def physics(tick, *, left=(), right=(), holders=(), z=0.8, x=0., in_target=False,
            open_left=0., native_success=False):
    tools = np.tile(np.eye(4), (2, 1, 1)).tolist()
    for i in range(2):
        tools[i][2][3] = 0.8
    return dict(kind="physics", tick=tick, sim_s=tick * .004, tcp_world=tools,
                block_pose_world=[x, 0., z, 1., 0., 0., 0.],
                box_functional_point_world=[0., 0., z], target_functional_point_world=[0., 0., z if in_target else 5.],
                actual_gripper_aperture=[open_left, 0.], native_success=native_success, contacts=[],
                contact_state=dict(holders=list(holders), contact_fingers={"left": list(left), "right": list(right)},
                                   releasing=[]))


class StagesTest(unittest.TestCase):
    def test_dual_contact_and_release_are_not_inferred_from_proximity(self):
        stages = Stages()
        stages.step(physics(0))
        stages.step(physics(1, left=(1,)))
        stages.step(physics(2, left=(1, 2), z=.86))
        result = stages.summary()["metrics"]
        self.assertEqual(result["left_approach"]["status"], "observed")
        self.assertEqual(result["left_bilateral"]["first"]["tick"], 2)
        self.assertEqual(result["left_grasp"]["status"], "not_observed")
        self.assertEqual(result["left_lift"]["status"], "not_observed")
        self.assertEqual(result["right_takeover"]["status"], "not_observed")
        self.assertEqual(result["ik_rejection"]["status"], "unavailable")

    def test_takeover_requires_40_ms_and_drop_requires_actual_fall(self):
        stages = Stages()
        rows = [physics(0), physics(1, left=(1, 2), holders=("left",), z=.86),
                physics(2, left=(1, 2), right=(3, 4), holders=("left", "right"), z=.86)]
        rows.extend(physics(tick, right=(3, 4), holders=("right",), z=.86, open_left=.95)
                    for tick in range(3, 14))
        for row in rows:
            stages.step(row)
        self.assertEqual(stages.summary()["metrics"]["right_takeover"]["first"]["tick"], 13)
        stages.step(physics(14, z=.856))
        self.assertEqual(stages.summary()["metrics"]["drop"]["status"], "not_observed")
        stages.step(physics(15, z=.82))
        self.assertEqual(stages.summary()["metrics"]["drop"]["first"]["tick"], 15)

    def test_task_predicate_independent_of_stage_proxy(self):
        stages = Stages()
        stages.step(physics(0))
        stages.step(physics(1, in_target=True, native_success=False))
        stages.step(physics(2, in_target=True, native_success=True))
        result = stages.summary()["metrics"]
        self.assertEqual(result["target_region"]["first"]["tick"], 1)
        self.assertEqual(result["native_success"]["first"]["tick"], 2)


class WireTest(unittest.TestCase):
    def test_rgb_is_not_transposed_or_reversed(self):
        rgb = np.zeros(SHAPE, dtype=np.uint8)
        rgb[0, 0] = [221, 19, 75]
        state = np.zeros(14)
        observation = {"joint_action": {"vector": state}, "observation": {n: {"rgb": rgb} for n in CAMERAS}}
        result = request_from(observation, 7, state)
        self.assertEqual(result["query_index"], 7)
        self.assertEqual(result["images"]["head_camera"][0, 0].tolist(), [221, 19, 75])
        with self.assertRaises(ValueError):
            request_from(observation, 7, state + .001)
        observation["observation"]["right_camera"]["rgb"] = rgb.transpose(2, 0, 1)
        with self.assertRaises(ValueError):
            request_from(observation, 7, state)

    def test_response_rejects_nonfinite_and_latent_shape(self):
        with self.assertRaises(ValueError):
            actions_from({"actions": np.zeros((50, 32))})
        response = np.zeros((50, 14))
        response[-1, -1] = np.nan
        with self.assertRaises(ValueError):
            actions_from({"actions": response})


class SummaryTest(unittest.TestCase):
    def test_fixture_and_censored_trials_have_no_formal_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for fixture in (True, False):
                campaign = {"plan": {"protocol": "test"}, "seeds": [0], "fixture": fixture}
                (root / "campaign.json").write_text(json.dumps(campaign))
                for variant in ("frozen", "lora"):
                    trial = root / f"seed-00-{variant}"
                    trial.mkdir(exist_ok=True)
                    stages = Stages()
                    row = physics(0)
                    stages.step(row)
                    (trial / "trajectory.jsonl").write_text(json.dumps(row) + "\n")
                    (trial / "receipt.json").write_text(json.dumps(dict(
                        schema="test", scene_seed=0, variant=variant,
                        source="protocol_fixture" if fixture else "step5_model", pair_match=True,
                        outcome="TASK_SUCCESS" if variant == "frozen" else "RESOURCE_CENSORED",
                        task_success=True if variant == "frozen" else None, analysis=stages.summary(),
                        physics_ticks=0, failure=None)))
                result = summarize(root)
                self.assertIsNone(result["formal_groups"]["frozen"]["success_rate"])
                self.assertEqual(result["pairs"][0]["excluded"],
                                 "PROTOCOL_FIXTURE" if fixture else "UNSCORED_TERMINATION")
