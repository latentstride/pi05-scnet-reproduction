"""Execute pinned upstream pure transform definitions only. Never load model params."""
import ast
import copy
import dataclasses
import hashlib
import json
import types
import unittest
from typing import Protocol, ClassVar, Sequence
import numpy as np
from observation_adapter import DATA, ROOT, EpisodeAdapter, check_observation, sha256


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('Missing upstream definitions')
    for node in selected:
        if isinstance(node,ast.ClassDef):
            node.bases = []  # DataTransformFn is a typing protocol, not runtime behavior.
    code = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(path),'exec'),namespace)
    return types.SimpleNamespace(**namespace)


def load_transforms():
    source = DATA/'stage3_sources/transforms.py'
    code = source.read_bytes()
    if hashlib.sha1(f'blob {len(code)}\0'.encode()+code).hexdigest() != '272375ea95a5e43a42a6c0ff14cf89d34da76a45':
        raise ValueError('Upstream transforms blob mismatch')
    return definitions(source,['DeltaActions','AbsoluteActions','Normalize','Unnormalize','make_bool_mask','pad_to_dim'],
                       {'dataclasses':dataclasses,'np':np})


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.tr = load_transforms()
        self.mask = self.tr.make_bool_mask(6,-1,6,-1)
        self.state = np.linspace(-0.5,0.8,14)
        self.actions = np.stack([self.state+0.2,self.state-0.1])
        self.actions[:,[6,13]] = [[0.25,0.75],[1,0]]

    def test_delta_absolute_and_grippers(self):
        data = {'state':self.state.copy(),'actions':self.actions.copy()}
        result = self.tr.DeltaActions(self.mask)(data)
        np.testing.assert_allclose(result['actions'][:,np.array(self.mask)],[[.2]*12,[-.1]*12])
        np.testing.assert_array_equal(result['actions'][:,[6,13]], self.actions[:,[6,13]])
        result = self.tr.AbsoluteActions(self.mask)(result)
        np.testing.assert_allclose(result['actions'],self.actions,atol=1e-15)
        np.testing.assert_array_equal(result['state'],self.state)

    def test_adapt_false_and_image_mapping(self):
        import einops
        path=DATA/'stage3_sources/policies_aloha_policy.py'
        code=path.read_bytes()
        self.assertEqual(hashlib.sha1(f'blob {len(code)}\0'.encode()+code).hexdigest(),'f16be3334584692ba505553220ab20ff7b2719c5')
        ap=definitions(path,['AlohaInputs','AlohaOutputs','_decode_aloha','_decode_state','_encode_actions','_encode_actions_inv'],
                       {'dataclasses':dataclasses,'np':np,'einops':einops,'ClassVar':ClassVar})
        with EpisodeAdapter('aloha-agilex','episode0') as adapter:
            obs=adapter.observation(100)
            raw=copy.deepcopy(obs)
            inputs=ap.AlohaInputs(adapt_to_pi=False)(obs)
            np.testing.assert_array_equal(inputs['state'],raw['state'])
            for dest,src in [('base_0_rgb','cam_high'),('left_wrist_0_rgb','cam_left_wrist'),('right_wrist_0_rgb','cam_right_wrist')]:
                np.testing.assert_array_equal(inputs['image'][dest],raw['images'][src].transpose(1,2,0))
                self.assertTrue(inputs['image_mask'][dest])
        output=ap.AlohaOutputs(adapt_to_pi=False)({'actions':np.pad(self.actions,((0,0),(0,18)))})
        self.assertEqual(output['actions'].shape,(2,14))
        np.testing.assert_array_equal(output['actions'],self.actions)

    def test_source_statistics_once(self):
        stats_path=ROOT/'research/gate25/pace_handover_block_assets_robotwin_norm_stats.json'
        manifest=json.loads((ROOT/'research/gate25/pace_handover_manifest.json').read_text())
        self.assertEqual(manifest['task'],'handover_block')
        expected=manifest['norm_stats_sha256']
        self.assertEqual(sha256(stats_path),expected)
        stats=json.loads(stats_path.read_text())['norm_stats']
        norm=self.tr.Normalize(None,use_quantiles=True)
        unnorm=self.tr.Unnormalize(None,use_quantiles=True)
        for key,raw in [('state',self.state),('actions',self.tr.DeltaActions(self.mask)({'state':self.state,'actions':self.actions.copy()})['actions'])]:
            st=types.SimpleNamespace(**{k:np.array(v) for k,v in stats[key].items()})
            self.assertEqual(st.q01.shape,(14,))
            normalized=norm._normalize_quantile(raw,st)
            restored=unnorm._unnormalize_quantile(normalized,st)
            np.testing.assert_allclose(restored,raw,atol=1e-12)
            # A second normalization is detectably wrong; adapter must retain raw values.
            self.assertFalse(np.allclose(normalized,norm._normalize_quantile(normalized,st)))
        policy=(DATA/'stage3_sources/policies_policy_config.py').read_text()
        self.assertEqual(policy.count('transforms.Normalize('),1)
        self.assertEqual(policy.count('transforms.Unnormalize('),1)
        cfg=(DATA/'stage3_sources/openpi_pace_configs.py').read_text()
        self.assertIn('adapt_to_pi=False',cfg)
        self.assertIn('use_delta_joint_actions=True',cfg)
        self.assertIn('"action_dim": 14',cfg)

    def test_reject_corrupt_observation(self):
        with EpisodeAdapter('arx-x5','episode0') as adapter:
            obs=adapter.observation(0)
            for value in [np.zeros(13,dtype=np.float32),np.full(14,np.nan,dtype=np.float32)]:
                broken=copy.deepcopy(obs); broken['state']=value
                with self.assertRaises(ValueError): check_observation(broken)
            broken=copy.deepcopy(obs); broken['images']['cam_high']=broken['images']['cam_high'].transpose(1,2,0)
            with self.assertRaises(ValueError): check_observation(broken)
            broken=copy.deepcopy(obs); broken['state'][6]=2
            with self.assertRaises(ValueError): check_observation(broken)
            with self.assertRaises(IndexError): adapter.observation(-1)
            self.assertNotIn('actions',obs)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TransformTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests_run':result.testsRun,'passed':result.wasSuccessful(),'failures':len(result.failures),
            'errors':len(result.errors),'parameters_loaded':False,'model_inference_tested':False,
            'scope':'AST-selected upstream definitions; no JAX/model/tokenizer runtime imports; Normalize arithmetic tested directly',
            'transforms_revision':'ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12',
            'aloha_config_revision':'ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12'}
    (DATA/'pace_transform_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
