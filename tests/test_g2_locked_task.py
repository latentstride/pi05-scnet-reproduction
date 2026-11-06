from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from g2_fixtures import ROOT, fixture
from g2_initial import digest_json
from g2_locked_score import collect
from g2_locked_task import load_trial, run_checked, verify_model_receipt
from summarize_experiments import LOCK_FIELDS
from test_g2_export import identity
from test_g2_integration import FixedPolicy, MockRobot


class LockedTrialTests(unittest.TestCase):
    def setup_trial(self, root, *, mismatch=False):
        adapter, qa, qr, bases = fixture()
        backend = MockRobot(adapter, qa, qr, bases)
        cfg = json.loads((ROOT/'research/g2/diagnostic-configs-20261105/A-native-w1.json').read_text())
        cfg.update(condition='Aloha-w1', max_actions=4, timeout_s=10.)
        cfg['reset_fixture'].update(formal_pair_id='synthetic_pair0', block_mass_kg=.01)
        episode = 'test-locked-episode'
        base_reset = backend.reset

        def reset(spec):
            return dict(base_reset(spec), reset_fixture_sha256=digest_json(cfg['reset_fixture']))

        backend.reset = reset
        expected = reset({'episode_id':episode, 'condition':cfg['condition']})
        if mismatch:
            expected['objects_task']['synthetic_block'][0] += .02
        meta = identity(dict(cfg, episode_id=episode), 'aloha-agilex', 'synthetic_transfer')
        meta['condition']['window_actions'] = 1
        meta['seed'] = cfg['scene_seed']
        config_path = root/'config.json'
        config_path.write_text(json.dumps(cfg))
        lock = root/'run_lock.json'
        lock.write_text(json.dumps(dict(schema_version=1,status='frozen',
            episodes=[{k:meta[k] for k in LOCK_FIELDS}],
            bindings={episode:dict(config_sha256=sha256(config_path.read_bytes()).hexdigest(),
                expected_initial=expected,trajectory_reference_id='synthetic_test')})))
        digest = sha256(lock.read_bytes()).hexdigest()

        @contextmanager
        def factory(spec):
            yield backend, None

        return lock, digest, episode, config_path, factory, FixedPolicy(qa)

    def test_locked_reset_gate_precedes_policy_and_no_synthetic_formal_file(self):
        for mismatch in (False,True):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lock,digest,episode,config_path,factory,policy = self.setup_trial(root,mismatch=mismatch)
                cfg,expected,meta = load_trial(lock,digest,episode,config_path)
                out = root/episode
                row = run_checked(cfg,expected,meta,factory,policy,out,verify_scene=lambda *_: {'synthetic':True})
                self.assertEqual(row['data_kind'],'diagnostic')
                self.assertFalse((out/'formal-record.json').exists())
                self.assertEqual(row['outcome'],'initial_state_mismatch' if mismatch else 'success')
                self.assertEqual(row['match_status'],'unmatched' if mismatch else 'matched')
                self.assertEqual(len(policy.inputs),0 if mismatch else 4)

    def test_changed_config_and_missing_results_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock,digest,episode,config_path,*_ = self.setup_trial(root)
            with config_path.open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError,'configuration hash'):
                load_trial(lock,digest,episode,config_path)
            with self.assertRaisesRegex(ValueError,'no terminal'):
                collect(lock,digest,root,root/'records.json')
            self.assertFalse((root/'records.json').exists())

    def test_scoring_schema_and_model_receipt_reject_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock,digest,episode,config_path,factory,policy = self.setup_trial(root)
            cfg,expected,meta = load_trial(lock,digest,episode,config_path)
            out = root/episode
            row = run_checked(cfg,expected,meta,factory,policy,out,verify_scene=lambda *_: {})
            # A transient schema fixture tests aggregation. The CPU entry emits no formal file.
            record = out/'formal-record.json'
            row['data_kind'] = 'formal'
            record.write_text(json.dumps(row))
            self.assertEqual(collect(lock,digest,root,root/'records.json')[0]['outcome'],'success')
            row['seed'] += 1
            record.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'locked roster'):
                collect(lock,digest,root,root/'rejected.json')
            model = root/'model.json'
            model.write_text(json.dumps(dict(revision='r', norm_stats_sha256='n',
                files=[dict(path='parameter',bytes=10,sha256='p')])))
            receipt = root/'startup.json'
            receipt.write_text(json.dumps(dict(status='ready',checkpoint_revision='r',
                checkpoint_files={'parameter':[10,0,'p']})))
            checkpoint = dict(revision='r',norm_stats_sha256='n',
                              manifest_sha256=sha256(model.read_bytes()).hexdigest())
            verify_model_receipt(checkpoint,model,receipt)
            checkpoint['norm_stats_sha256']='other'
            with self.assertRaisesRegex(ValueError,'locked checkpoint'):
                verify_model_receipt(checkpoint,model,receipt)
