import json
from pathlib import Path
import subprocess
import sys
import unittest

from cuda_wait_probe import compare


class WaitProbeTests(unittest.TestCase):
    def rows(self):
        control = dict(device=1, gpu_uuid='GPU-fixed', flags={'before':0,'after':0}, output_sha256='a'*64, shape=[2048,2048], dtype='bfloat16',
                       policy='control', process_cpu_seconds=4.9, wall_seconds=5)
        return control, dict(control, policy='blocking', process_cpu_seconds=.1, flags={'before':0,'after':4})

    def test_compare_requires_same_output_device_shape_and_dtype(self):
        control, candidate = self.rows()
        result=compare(control,candidate)
        self.assertTrue(result['numerical_equal'])
        self.assertAlmostEqual(result['control_cpu_per_wall'],.98)
        self.assertAlmostEqual(result['candidate_cpu_per_wall'],.02)
        for key in ('device','gpu_uuid','output_sha256','shape','dtype','policy'):
            with self.assertRaises(ValueError): compare(control,dict(candidate,**{key:'wrong'}))

    def test_already_blocking_control_is_inconclusive(self):
        control,candidate=self.rows()
        control['flags']={'before':4,'after':4}
        self.assertFalse(compare(control,candidate)['intervention'])
        control['flags']={'before':0,'after':0}
        candidate['flags']={'before':0,'after':0}
        with self.assertRaises(ValueError):compare(control,candidate)

    def test_default_entrypoint_is_read_only_even_with_gpu_packages_absent(self):
        result=subprocess.run([sys.executable,'-S',str(Path(__file__).with_name('cuda_wait_probe.py'))],
                              check=True,capture_output=True,text=True)
        plan=json.loads(result.stdout)
        self.assertFalse(plan['gpu_execution'])
        self.assertTrue(plan['external_supervisor_required'])
        self.assertEqual(plan['seconds'],5)


if __name__ == '__main__': unittest.main()
