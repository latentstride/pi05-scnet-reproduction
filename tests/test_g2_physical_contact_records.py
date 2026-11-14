"""Retain measured PhysX contact evidence, independent of holder classification."""
from types import SimpleNamespace as NS
import unittest
import numpy as np
import g2_robotwin_backend as backend


class PhysicalContactRecordTest(unittest.TestCase):
    def test_contact_ids_positions_impulses_and_signed_separation_are_preserved(self):
        contact=NS(bodies=[NS(entity=NS(name='box',per_scene_id=7)),
                           NS(entity=NS(name='link8',per_scene_id=22))],
                   points=[NS(position=np.array([1.,2.,3.]),
                              impulse=np.array([.01,0.,0.]),separation=-.0003)])
        scene=NS(get_contacts=lambda:[contact])
        self.assertTrue(hasattr(backend,'physical_contact_records'))
        rows=backend.physical_contact_records(scene)
        self.assertEqual(rows[0]['body_ids'],[7,22])
        self.assertEqual(rows[0]['points'][0]['position_m'],[1.,2.,3.])
        self.assertEqual(rows[0]['points'][0]['impulse_Ns'],[.01,0.,0.])
        self.assertEqual(rows[0]['points'][0]['separation_m'],-.0003)
