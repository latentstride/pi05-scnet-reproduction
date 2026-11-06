import io
import pickle
import unittest
from pathlib import Path
import numpy as np
from g2_audit_expert_metadata import NumpyOnlyUnpickler, timeline, spawn

ROOT = Path(__file__).resolve().parents[1]

class ExpertMetadataTests(unittest.TestCase):
    def test_three_official_episodes_match_every_target(self):
        import h5py
        for episode, frames, last_tick in [(0,295,3945),(1,277,3655),(2,282,3764)]:
            with self.subTest(episode=episode):
                path=ROOT/f'research/g2/expert-cache-uncached-20261106/episode{episode}.pkl'
                with path.open('rb') as stream: cache=NumpyOnlyUnpickler(stream).load()
                with h5py.File(ROOT/f'research/robotwin/data_probe/aloha-agilex/episode{episode}.hdf5') as f:
                    actual=np.asarray(f['joint_action/vector'])
                predicted,ticks,stages=timeline(cache,actual[0])
                np.testing.assert_array_equal(predicted,actual)
                self.assertEqual(len(ticks),frames); self.assertEqual(ticks[-1],last_tick)
                self.assertTrue(np.all(np.diff(ticks)>=0)); self.assertIn(0,np.diff(ticks))
                self.assertEqual(sum(s['last_tick']-s['first_tick'] for s in stages if s['kind'] in ('LG','RG')),1200)

    def test_unexpected_pickle_global_rejected_without_execution(self):
        with self.assertRaises(pickle.UnpicklingError):
            NumpyOnlyUnpickler(io.BytesIO(pickle.dumps(Path('/tmp/unused')))).load()

    def test_seed_reconstruction_does_not_mutate_global_rng(self):
        np.random.seed(42); before=np.random.get_state(); first=spawn(0); after=np.random.get_state()
        self.assertEqual(first,spawn(0)); self.assertNotEqual(first,spawn(1))
        np.testing.assert_array_equal(before[1],after[1]); self.assertEqual(before[2:],after[2:])
