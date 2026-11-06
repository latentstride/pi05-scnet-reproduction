from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import json
import tempfile
import unittest

import numpy as np

from g2_fixtures import ROOT, fixture
from g2_collision import JointCollisionWorld
from g2_initial import compare_initial, compare_objects
from g2_run_task import execute
from g2_task_runtime import validate_config
from test_g2_integration import FixedPolicy, MockRobot, spec
from g2_episode import ExpertWaypoints, run_episode
from g2_robotwin_backend import joint_space_setup


def configuration():
    cfg=json.loads((ROOT / "research/g2/diagnostic-configs-20261105/B-arx-w1.json").read_text())
    cfg.update(data_kind="diagnostic",episode_id="entry-mock",max_actions=4,timeout_s=5.)
    return cfg


class RuntimeTests(unittest.TestCase):
    def test_same_entry_factory_for_two_embodiments_and_windows(self):
        for robot in ("aloha-agilex","ARX-X5"):
            for window in (1,4):
                with self.subTest(robot=robot,window=window), tempfile.TemporaryDirectory() as temp:
                    config=configuration()
                    config.update(embodiment=robot,window=window,
                                  condition="Aloha" if robot=="aloha-agilex" else "ARX")
                    adapter,qa,qr,bases=fixture()
                    backend=MockRobot(adapter,qa,qr,bases)
                    closed=[]
                    @contextmanager
                    def factory(spec):
                        try:
                            yield backend,adapter if robot=="ARX-X5" else None
                        finally:
                            closed.append(True)
                    result=execute(config,factory,FixedPolicy(qa),Path(temp)/"episode")
                    self.assertEqual(result["outcome"],"success",result)
                    self.assertEqual(result["queries"],4//window)
                    self.assertEqual(closed,[True])
                    self.assertTrue(backend.closed)

    def test_scene_setup_failure_is_a_complete_episode_record(self):
        with tempfile.TemporaryDirectory() as temp:
            config=configuration()
            adapter,qa,qr,bases=fixture()
            @contextmanager
            def factory(spec):
                raise RuntimeError("injected real scene setup failure")
                yield
            result=execute(config,factory,FixedPolicy(qa),Path(temp)/"failed")
            self.assertEqual(result["outcome"],"environment_error")
            self.assertIn("real scene setup",result["failure"]["detail"])
            self.assertTrue((Path(temp)/"failed/result.json").exists())

    def test_expert_never_requires_policy_proxy_ik(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter,qa,qr,bases=fixture()
            adapter.observe=lambda *args: (_ for _ in ()).throw(AssertionError("proxy IK was called"))
            adapter.source_collision=lambda q,g: False
            rows=np.tile(np.column_stack((qa,[.5,.5])).ravel(),(4,1))
            result=run_episode(spec(4),MockRobot(adapter,qa,qr,bases),None,Path(temp)/"expert",
                               adapter=adapter,expert=ExpertWaypoints(rows,"synthetic expert"))
            self.assertEqual(result["outcome"],"success",result)

    def test_bootstrap_gripper_path_uses_declared_native_endpoints(self):
        adapter,qa,qr,bases=fixture()
        values={}
        def joint(name):
            return NS(get_name=lambda:name,set_drive_target=lambda value:values.__setitem__(name,value),
                      set_drive_velocity_target=lambda value:None)
        class Robot:
            is_dual_arm=False
            def set_planner(self,scene=None):
                pass
            def set_gripper(self,*args):
                raise AssertionError("unbounded upstream mapping")
        original=Robot.set_gripper
        robot=Robot()
        robot.left_gripper=[(joint(name),1,0) for name in adapter.arx[0].grip.names]
        robot.right_gripper=robot.left_gripper
        with joint_space_setup(Robot,[a.grip for a in adapter.arx]):
            robot.set_planner()
            profile=robot.left_planner.plan_grippers(0.,1.)
            self.assertEqual(profile["num_step"],200)
            robot.set_gripper(1.,"left")
            self.assertTrue(all(value==.04 for value in values.values()))
        self.assertIs(Robot.set_gripper,original)

    def test_shared_collision_model_is_updated_atomically_and_restored(self):
        names=[f"j{i}" for i in range(16)]
        class Model:
            q=np.zeros(16)
            def get_qpos(self):return self.q.copy()
            def set_qpos(self,q,full):self.q=q.copy()
        model=Model()
        inspected=[]
        class World:
            def check_collision(self):
                inspected.append(model.q.copy())
                return []
        checker=JointCollisionWorld(World(),[model,model],[names,names],
                                    [tuple(names[:6]),tuple(names[8:14])],
                                    [tuple(names[6:8]),tuple(names[14:16])])
        self.assertTrue(checker(np.ones((2,6)),np.full((2,2),2.)))
        np.testing.assert_array_equal(inspected[0],np.r_[np.ones(6),[2,2],np.ones(6),[2,2]])
        np.testing.assert_array_equal(model.q,np.zeros(16))
        def fail():raise RuntimeError("FCL failure")
        checker.world.check_collision=fail
        with self.assertRaises(RuntimeError):checker(np.ones((2,6)),np.ones((2,2)))
        np.testing.assert_array_equal(model.q,np.zeros(16))

    def test_diagnostic_configuration_cannot_enable_untested_contact_exemptions(self):
        config=configuration()
        config["contact_mode"]="ignore_task_object"
        with self.assertRaisesRegex(ValueError,"contact"):validate_config(config)
        config=configuration()
        config["data_kind"]="formal"
        with self.assertRaisesRegex(ValueError,"diagnostic"):validate_config(config)

    def test_initial_matching_compares_actual_objects_and_quaternion_equivalence(self):
        adapter,qa,qr,bases=fixture()
        backend=MockRobot(adapter,qa,qr,bases)
        actual=backend.reset(spec(1))
        expected=deepcopy(actual)
        expected["objects_task"]["synthetic_block"][3:]=[-1,0,0,0]
        self.assertEqual(compare_initial(actual,expected),[])
        expected["objects_task"]["synthetic_block"][0]+=.02
        self.assertEqual(compare_objects(actual,expected),["synthetic_block"])
        self.assertIn("object_synthetic_block",compare_initial(actual,expected))

    def test_mechanical_endpoint_configuration_is_derived_not_fixture_copied(self):
        config=configuration()
        self.assertEqual(config["calibration"]["ARX-X5"]["left"]["closed_m"],[0.,0.])
        self.assertEqual(config["calibration"]["ARX-X5"]["left"]["opened_m"],[.044,.044])
        evidence=config["endpoint_evidence"]["ARX-X5"]["left"]
        self.assertEqual(evidence["configured_closed_m"],-.01)
        self.assertEqual(evidence["configured_open_m"],.045)
        self.assertFalse(config["calibration"]["ARX-X5"]["left"]["contact_dynamics_verified"])

class CollisionPolicyTests(unittest.TestCase):
    def test_fixed_self_profile_does_not_exempt_other_arm_or_environment(self):
        from g2_collision import audited_arx_self_profile
        from g2_fixtures import ASSETS
        profile=audited_arx_self_profile(ASSETS)
        class Model:
            def __init__(self,name):self.name,self.q=name,np.zeros(8)
            def get_name(self):return self.name
            def get_qpos(self):return self.q.copy()
            def set_qpos(self,q,full):self.q=q.copy()
        def row(a,linka,b,linkb):
            return NS(collision_type="test",object_name1=a,link_name1=linka,object_name2=b,link_name2=linkb)
        collisions=[row("L","base_link","L","link2"),row("L","base_link","R","link2"),
                    row("L","base_link","table","table"),row("other","base_link","other","link2")]
        world=NS(check_collision=lambda:collisions)
        names=[f"j{i}" for i in range(8)]
        checker=JointCollisionWorld(world,[Model("L"),Model("R")],[names,names],
                                    [tuple(names[:6])]*2,[tuple(names[6:])]*2)
        checker.use_self_profile(profile)
        self.assertFalse(checker(np.zeros((2,6)),np.zeros((2,2))))
        self.assertEqual(len(checker.last_raw_collisions),4)
        self.assertEqual(len(checker.last_ignored_collisions),1)
        self.assertEqual(len(checker.last_collisions),3)
