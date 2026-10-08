"""CPU tests for diagnostic controls and coordinate reporting, no GPU/model load."""
import json
from pathlib import Path
import unittest

import torch

from diagnose_block7 import details, rowwise, validate_controls


class DiagnosticTests(unittest.TestCase):
    def test_saved_real_weight_controls_retain_original_failures(self):
        for backend in ("default", "math"):
            with self.subTest(backend=backend):
                report=Path(__file__).parent/"evidence-block7"/f"{backend}-report.json"
                validate_controls(json.loads(report.read_text()))

    def test_padded_shape_control_preserves_rows_and_operator_geometry(self):
        value=torch.arange(16).reshape(1,8,2).float()
        # Simulate a shape-sensitive operator; test is about the control geometry.
        def operator(x):return x*2+x.shape[1]
        expected=operator(value)
        self.assertFalse(torch.equal(rowwise(operator,value,True),expected))
        torch.testing.assert_close(rowwise(operator,value,'padded'),expected,rtol=0,atol=0)

    def test_max_normalized_coordinate_is_not_max_absolute_coordinate(self):
        reference=torch.tensor([[[229.69900512695312,.0551152229309082]]])
        actual=torch.tensor([[[229.6988067626953,.05514192581176758]]])
        result=details(actual,reference)
        self.assertEqual(result['worst_flat_index'],0)
        self.assertEqual(result['max_normalized_coordinate']['coordinate'],[0,0,1])
        self.assertEqual(result['violations'],1)
        self.assertEqual(result['failing_coordinates'][0]['coordinate'],[0,0,1])
        self.assertGreater(result['failing_coordinates'][0]['error'],result['failing_coordinates'][0]['bound'])


if __name__=='__main__':unittest.main()
