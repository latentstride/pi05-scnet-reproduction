"""The CPU diagnostic must retain an actual failed phase and selected rows."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


TOOL=Path(__file__).resolve().parents[1]/'scripts/g2_arx_endpoint_cpu.py'


class EndpointDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('endpoint_under_test',TOOL)
        cls.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_selected_failure_row_is_not_replaced_by_fixed_query_two_rows(self):
        chunk=np.zeros((50,14))
        chunk[35,6]=.8
        rows=self.module.select_rows(chunk,[35,36,50])
        self.assertEqual([n for n,_ in rows],[35,36,50])
        self.assertEqual(rows[1][1][6],.8)
        with self.assertRaises(ValueError):self.module.select_rows(chunk,[0])
        with self.assertRaises(ValueError):self.module.select_rows(chunk,[51])

    def test_unknown_penetration_is_not_reported_as_clear_or_a_number(self):
        self.assertIsNone(self.module.depth_mm(None))
        self.assertEqual(self.module.depth_mm(.003),3.)

    def test_cpu_build_failure_writes_phase_and_exits_nonzero(self):
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/'evidence'
            with patch.object(self.module,'build_scene',side_effect=RuntimeError('CPU scene unavailable')):
                status=self.module.diagnose({'chunk':np.zeros((50,14)).tolist(),
                    'job':'recorded','query':1},Path('audit'),Path('assets'),output,[36])
            self.assertEqual(status,1)
            receipt=json.loads((output/'receipt.json').read_text())
            self.assertEqual(receipt['status'],'error')
            self.assertEqual(receipt['failed_phase'],'build_scene')
            self.assertIn('CPU scene unavailable',receipt['error'])


if __name__=='__main__':unittest.main()
