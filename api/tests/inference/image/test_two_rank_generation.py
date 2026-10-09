import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from api.inference.image.two_rank_generation import TwoRankQwenImage
from api.inference.image.qwen_image_pipeline import GIB
from api.inference.resources import ResourceManager, ResourceExhausted
from api.services.image_jobs import ImageJobs


class DualAdmissionTests(unittest.TestCase):
    def test_default_backend_remains_single_and_requires_explicit_opt_in(self):
        with patch.dict('os.environ',{},clear=True):
            self.assertEqual(ImageJobs._configuration(),('auto','sequential'))
        with patch.dict('os.environ',{'KADAN_IMAGE_BACKEND':'dual'}):
            self.assertEqual(ImageJobs._configuration(),('dual','component'))

    def test_unsupported_scope_is_rejected_before_resource_allocation(self):
        for aspect,count,image in [('16:9',1,None),('1:1',2,None),('1:1',1,'edit')]:
            with self.assertRaises(ValueError):TwoRankQwenImage.validate('prompt',aspect,count,image)
        with self.assertRaises(ValueError):TwoRankQwenImage.validate('x'*2001,'1:1',1)
        TwoRankQwenImage.validate('prompt','1:1',1)

    def test_individual_gpu_and_host_plans_do_not_pool_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'weights.safetensors').write_bytes(b'fixture')
            for host,devices in [(300*GIB,{0:40*GIB}), (300*GIB,{0:40*GIB,1:20*GIB}), (8*GIB,{0:40*GIB,1:40*GIB})]:
                resources=ResourceManager(host,devices)
                with self.assertRaises(ResourceExhausted):TwoRankQwenImage(path,resources,devices=[0,1])
                self.assertEqual(resources.snapshot()['reservations'],{})
