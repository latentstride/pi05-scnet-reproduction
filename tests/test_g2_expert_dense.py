import ast,json,unittest
from pathlib import Path
import tempfile
import numpy as np
import h5py
from g2_audit_expert_metadata import NumpyOnlyUnpickler
from g2_expert_dense import dense,packed_controls
from g2_native_expert_cpu import unpack_trace

ROOT=Path(__file__).resolve().parents[1]
class ExpertDenseTests(unittest.TestCase):
    def test_every_dense_trace_agrees_with_all_saved_commands_and_velocity_paths(self):
        for ep,last in [(0,3945),(1,3655),(2,3764)]:
            with self.subTest(episode=ep):
                with (ROOT/f'research/g2/expert-cache-uncached-20261106/episode{ep}.pkl').open('rb') as f:paths=NumpyOnlyUnpickler(f).load()
                with h5py.File(ROOT/f'research/robotwin/data_probe/aloha-agilex/episode{ep}.hdf5') as f:observed=f['joint_action/vector'][:]
                ticks=np.load(ROOT/f'research/g2/expert-metadata-20261106/episode{ep}_derived_timing.npz')['physics_tick']
                data,segments=dense(paths,observed[0]);self.assertEqual(len(data['target']),last+1)
                np.testing.assert_array_equal(data['target'][ticks],observed)
                for seg in segments:
                    if seg['kind']=='L':
                        start,end=seg['first_tick'],seg['last_tick'];native=paths['left_joint_path'][seg['path_index']]
                        np.testing.assert_array_equal(data['arm_velocity'][start+1:end+1,0],native['velocity'])
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/'runtime.npz';np.savez_compressed(p,**packed_controls(dict(data,frame_ticks=ticks)))
                    unpacked=unpack_trace(p)
                    np.testing.assert_array_equal(unpacked['target'],data['target'])
                    np.testing.assert_array_equal(unpacked['arm_velocity'],data['arm_velocity'])
    def test_source_actor_wrapper_mass_is_ten_grams(self):
        source=ROOT/'research/g2/expert-source-20261105/envs/utils/actor_utils.py'
        tree=ast.parse(source.read_text());actor=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Actor')
        init=next(n for n in actor.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        self.assertEqual(ast.literal_eval(init.args.defaults[-1]),.01)
        import inspect
        from g2_native_expert_cpu import run
        self.assertEqual(inspect.signature(run).parameters['mass'].default,.01)
    def test_gripper_updates_match_pinned_original_function(self):
        from types import SimpleNamespace as NS
        class Joint:
            def __init__(self):self.target=0.;self.velocity=0.
            @property
            def drive_target(self):return np.array([self.target])
            def get_drive_target(self):return self.drive_target
            def set_drive_target(self,v):self.target=float(np.asarray(v).ravel()[0])
            def set_drive_velocity_target(self,v):self.velocity=float(np.asarray(v).ravel()[0])
        source=ast.parse((ROOT/'research/g2/expert-source-20261105/envs/robot/robot.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Robot')
        original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='set_gripper')
        ns={'np':np};exec(compile(ast.Module(body=[original],type_ignores=[]),'pinned_set_gripper','exec'),ns)
        original=ns['set_gripper'];old_fingers=[[Joint(),Joint()],[Joint(),Joint()]]
        obj=NS(left_entity=None,right_entity=None,left_gripper=[(j,1.,0.) for j in old_fingers[0]],right_gripper=[(j,1.,0.) for j in old_fingers[1]],left_gripper_scale=[-.01,.045],right_gripper_scale=[-.01,.045],_entity_qf=lambda entity:None)
        obj.get_normal_real_gripper_val=lambda:np.clip([(g[0].target+.01)/.055 for g in old_fingers],0,1)
        tree=ast.parse((ROOT/'scripts/g2_native_expert_cpu.py').read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        replacement=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='grip')
        fingers=[[Joint(),Joint()],[Joint(),Joint()]];normalized=np.zeros(2)
        ns2={'np':np,'fingers':fingers,'normalized':normalized};exec(compile(ast.Module(body=[replacement],type_ignores=[]),'replay_gripper','exec'),ns2)
        replay=ns2['grip']
        for a,b,eps in [(0.,1.,.005),(1.,0.,-.005),(0.,1.,.005)]:
            for value in np.r_[np.linspace(a,b,200),np.repeat(b,100)]:
                for side in range(2):
                    original(obj,value,('left','right')[side],eps);replay(side,value,eps)
                    for orig,got in zip(old_fingers[side],fingers[side]):
                        self.assertEqual(orig.target,got.target);self.assertEqual(orig.velocity,got.velocity)
        from g2_settle_expert_cpu import run as settle_run
        import inspect
        self.assertEqual(inspect.signature(settle_run).parameters['block_mass_kg'].default,.01)
