from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import numpy as np

from g2_fixtures import robot_arms
from g2_initial import capture_initial
from g2_native_backend import NativeBackend
from g2_episode import run_episode
from test_g2_integration import FixedPolicy,spec
from g2_adapter import G2Error


def native_fixture(root,fault=None):
    arms=robot_arms("aloha-agilex")
    names=arms[0].chain.names+arms[0].grip.names+arms[1].chain.names+arms[1].grip.names
    q=np.zeros(16)
    q[[6,7,14,15]]=.02
    state={"actual":q.copy(),"target":q.copy(),"gripper":[.5,.5],"velocity":np.ones(16)}
    joints=[]
    limits=[*arms[0].chain.limits,*arms[0].grip.limits,*arms[1].chain.limits,*arms[1].grip.limits]
    for index,name in enumerate(names):
        joints.append(NS(get_name=lambda name=name:name,
                         get_limits=lambda index=index:np.array([limits[index]]),
                         get_drive_target=lambda index=index:np.array([state["target"][index]]),
                         set_drive_target=lambda value,index=index:state["target"].__setitem__(index,value),
                         set_drive_velocity_target=lambda value,index=index:state["velocity"].__setitem__(index,value)))
    entity=NS(get_active_joints=lambda:joints,get_qpos=lambda:state["actual"].copy(),
              get_root_pose=lambda:NS(p=np.zeros(3),q=np.array([1,0,0,0])),
              get_name=lambda:"aloha",compute_passive_force=lambda **kwargs:np.zeros(16),set_qf=lambda value:None)
    def vector(actual,side):
        start=0 if side==0 else 8
        return list(state["actual" if actual else "target"][start:start+6])+[state["gripper"][side]]
    robot=NS(left_entity=entity,right_entity=entity,left_arm_joints=joints[:6],right_arm_joints=joints[8:14],
             get_left_arm_jointState=lambda:vector(False,0),get_right_arm_jointState=lambda:vector(False,1),
             get_left_arm_real_jointState=lambda:vector(True,0),get_right_arm_real_jointState=lambda:vector(True,1))
    actor=NS(get_pose=lambda:NS(p=np.zeros(3),q=np.array([1,0,0,0])))
    physical=NS(step=lambda:state.__setitem__("actual",state["target"].copy()),get_timestep=lambda:.004)
    env=NS(robot=robot,scene=physical,box=actor,target_box=actor,table=actor,take_action_cnt=0,step_lim=500)
    def observation():
        left,right=vector(False,0),vector(False,1)
        return dict(observation={camera:{"rgb":np.full((8,8,3),env.take_action_cnt,dtype=np.uint8)}
                                 for camera in ("head_camera","left_camera","right_camera")},
                    joint_action=dict(vector=np.array(left+right,dtype=np.float32),left_arm=left[:6],left_gripper=left[6],
                                      right_arm=right[:6],right_gripper=right[6]))
    env.get_obs=observation
    env.get_instruction=lambda:"native synthetic"
    env.check_success=lambda:env.take_action_cnt>=4
    closed=[]
    env.close_env=lambda **kwargs:closed.append(True)
    def take_action(action,action_type):
        if action_type!="qpos":raise AssertionError("native path changed")
        env.take_action_cnt+=1
        row=action.reshape(2,7)
        for side,offset in ((0,0),(1,8)):
            state["target"][offset:offset+6]=row[side,:6]
            state["gripper"][side]=row[side,6]
            state["target"][offset+6:offset+8]=row[side,6]*.04
        env.scene.step()
        if fault=="control":raise RuntimeError("injected native controller fault")
    env.take_action=take_action
    class Video:
        path=root/"synthetic-video"
        def frame(self,observation):pass
        def close(self):pass
    initial=capture_initial(env,arms,np.eye(4),"native","0"*64,"synthetic-fixture",True)
    return NativeBackend(env,arms,np.eye(4),initial,Video()),state,closed,physical


class NativeBackendTests(unittest.TestCase):
    def test_complete_native_episodes_keep_qpos_and_true_scene_clock(self):
        for window in (1,4):
            with self.subTest(window=window),tempfile.TemporaryDirectory() as temp:
                backend,state,closed,physical=native_fixture(Path(temp))
                config=spec(window,"Aloha")
                result=run_episode(config,backend,FixedPolicy(np.zeros((2,6))),Path(temp)/"episode")
                self.assertEqual(result["outcome"],"success",result)
                self.assertEqual(result["queries"],4//window)
                self.assertEqual(backend.tick,4)
                self.assertEqual(result["control_calls"],4)
                self.assertIs(backend.env.scene,physical)
                self.assertEqual(closed,[True])
                self.assertIn('"sim_s": 0.016',(Path(temp)/"episode/events.jsonl").read_text())

    def test_native_partial_fault_holds_and_closes(self):
        with tempfile.TemporaryDirectory() as temp:
            backend,state,closed,physical=native_fixture(Path(temp),"control")
            result=run_episode(spec(4,"Aloha"),backend,FixedPolicy(np.zeros((2,6))),Path(temp)/"episode")
            self.assertEqual(result["outcome"],"control_error")
            self.assertEqual(backend.tick,1)
            self.assertTrue(np.all(state["velocity"]==0))
            self.assertEqual(closed,[True])
            self.assertIs(backend.env.scene,physical)

    def test_invalid_native_prefix_is_rejected_before_any_physics_step(self):
        with tempfile.TemporaryDirectory() as temp:
            backend,state,closed,physical=native_fixture(Path(temp))
            backend.reset(spec(1,"Aloha"))
            bad=np.zeros((1,14))
            bad[0,6]=1.1
            with self.assertRaises(G2Error):backend.execute(bad,lambda event:None,lambda:1)
            self.assertEqual(backend.tick,0)
            backend.close(Path(temp))
