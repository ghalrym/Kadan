"""Recovery contract tests; no Docker daemon or GPU is accessed."""
import copy
import unittest
from unittest.mock import patch

import launch_probe


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.before = dict(Id='original-id', Image='sha256:original',
            Config={'Env': ['KEY=original']}, HostConfig={'DeviceRequests': ['GPU1']},
            Mounts=[{'Source': '/original', 'Destination': '/app/api'}],
            Path='python', Args=['-m', 'uvicorn'], State={'Running': True})

    def test_starts_exact_container_and_ignores_changed_compose_files(self):
        stopped = copy.deepcopy(self.before)
        stopped['State']['Running'] = False
        with patch.object(launch_probe, 'inspect_container', side_effect=[stopped, self.before]) as inspect, \
                patch.object(launch_probe, 'run') as run:
            launch_probe.restore_container(self.before)
        run.assert_called_once_with('docker', 'start', 'original-id', timeout=45)
        self.assertEqual(inspect.call_args_list[0].args, ('original-id',))
        self.assertEqual(inspect.call_args_list[1].args, ('original-id',))

    def test_refuses_changed_container_configuration_before_start(self):
        for key, changed in [('Id', 'replacement-id'), ('Image', 'sha256:replacement'),
                ('Config', {'Env': ['KEY=changed']}), ('HostConfig', {'DeviceRequests': ['GPU0']})]:
            with self.subTest(key=key):
                stopped = copy.deepcopy(self.before)
                stopped[key] = changed
                with patch.object(launch_probe, 'inspect_container', return_value=stopped), \
                        patch.object(launch_probe, 'run') as run:
                    with self.assertRaisesRegex(AssertionError, 'identity/config changed'):
                        launch_probe.restore_container(self.before)
                    run.assert_not_called()

    def test_checks_identity_and_running_state_after_start(self):
        for update in ({'Id': 'replacement'}, {'State': {'Running': False}}):
            with self.subTest(update=update):
                restored = self.before | update
                with patch.object(launch_probe, 'inspect_container', side_effect=[self.before, restored]), \
                        patch.object(launch_probe, 'run'):
                    with self.assertRaises(AssertionError):
                        launch_probe.restore_container(self.before)


if __name__ == '__main__':
    unittest.main()
