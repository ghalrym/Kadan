import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('preview', Path(__file__).with_name('preview_test_run.py'))
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class PreviewTests(unittest.TestCase):
    def args(self):
        return dict(image='sha256:' + 'e' * 64, state_dir='/tmp/test-state', checkpoint_dir='/models/checkpoint',
                    checkpoint_relative='snapshots/small-test', env_file='/tmp/private-test.env',
                    network='kadan-native-review-test', gpu=1, host_bytes=4 * 1024**3,
                    device_bytes=20 * 1024**3, port=18000)

    def test_isolated_command(self):
        result = preview.command(**self.args())
        self.assertIn('127.0.0.1:18000:8000', result)
        self.assertIn('device=1', result)
        self.assertIn('KADAN_GPU=0', result)
        self.assertIn('KADAN_NATIVE_WORKER=/opt/kadan/bin/run-model-worker', result)
        self.assertIn('type=bind,src=/models/checkpoint,dst=/var/lib/kadan/models/snapshots/small-test,readonly', result)
        self.assertNotIn('--privileged', result)
        self.assertIn('--read-only', result)
        self.assertEqual(result[-1], self.args()['image'])

    def test_rejects_unsafe_or_ambiguous_values(self):
        for key, value in [('image', 'latest'), ('network', 'host'), ('network', 'bridge'),
                           ('state_dir', '/'), ('state_dir', '/models'), ('checkpoint_relative', '../models'),
                           ('checkpoint_relative', '/models'), ('checkpoint_dir', '/models,x'),
                           ('gpu', -1), ('host_bytes', 1), ('device_bytes', 1), ('port', 80)]:
            args = self.args()
            args[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                preview.command(**args)


if __name__ == '__main__':
    unittest.main()
