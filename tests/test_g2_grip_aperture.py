"""Independent-finger geometry reduction; measured state is never overwritten."""
from dataclasses import replace
import json
import unittest
import numpy as np
from g2_adapter import G2Error
from g2_fixtures import fixture, snapshot, ASSETS, AUDIT, ROOT
from g2_assets import load_arms

class ApertureTests(unittest.TestCase):
    def test_asymmetric_equal_aperture_preserves_center_and_residual(self):
        grip=replace(fixture()[0].arx[0].grip,state_mode='independent_aperture_v1',state_source='synthetic test')
        native=np.array([.02883804589509964,.030293013900518417]);before=native.copy()
        x=grip.measure(native)
        self.assertAlmostEqual(x['normalized'],sum(native)/.08)
        self.assertAlmostEqual(x['center_shift_from_symmetric_m'],(native[0]-native[1])/2)
        self.assertAlmostEqual(sum(x['reduction_residual_m']),0.)
        np.testing.assert_array_equal(native,before)
        for bad in ([float('nan'),.02],[-.01,.02],[.02,.06]):
            with self.assertRaises((ValueError,G2Error)):grip.read(bad)
        self.assertAlmostEqual(grip.read([0.,.04]),.5)

    def test_unequal_calibrated_travel_uses_total_aperture(self):
        base=fixture()[0].arx[0].grip
        grip=replace(base,opened=np.array([.03,.04]),state_mode='independent_aperture_v1',state_source='test')
        self.assertAlmostEqual(grip.read([.02,.03]),.05/.07)
        with self.assertRaises(G2Error):grip.read([.033,.02])

    def test_explicit_profile_and_hashed_urdf_load(self):
        c=json.loads((ROOT/'research/g2/contact-configs-stage5-20261105/B-arx-w1-contact.json').read_text())
        arms=load_arms(ASSETS,AUDIT,'ARX-X5',c['calibration']['ARX-X5'])
        x=arms[0].grip.measure([.02883804589509964,.030293013900518417])
        self.assertAlmostEqual(x['normalized'],.6719438613)
        self.assertEqual(x['state_mode'],'independent_aperture_v1')
        self.assertEqual(arms[0].grip.tolerance,.001)

    def test_next_policy_observation_uses_measured_aperture(self):
        adapter,qa,qr,bases=fixture()
        adapter.arx=tuple(replace(a,grip=replace(a.grip,state_mode='independent_aperture_v1',state_source='test')) for a in adapter.arx)
        snap=snapshot(adapter,qr,bases)
        native=np.array([.02883804589509964,.030293013900518417])
        sample=snap.arms[0];qpos=sample.qpos.copy()
        for name,value in zip(adapter.arx[0].grip.names,native):qpos[sample.names.index(name)]=value
        snap=replace(snap,arms=(replace(sample,qpos=qpos),snap.arms[1]))
        result=adapter.observe(snap,'synthetic handover',3.)
        self.assertTrue(result.ok,result.detail)
        self.assertEqual(result.payload['state'].shape,(14,))
        self.assertAlmostEqual(float(result.payload['state'][6]),sum(native)/.08,places=6)
        log=result.diagnostic['grip_measurement'][0]
        np.testing.assert_array_equal(log['native_m'],native)
        self.assertGreater(abs(log['center_shift_from_symmetric_m']),.0007)
        np.testing.assert_array_equal(snap.arms[0].qpos,qpos)
