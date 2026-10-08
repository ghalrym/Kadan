"""Transport tests replace every subprocess; no container operations occur."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from native.tests.reference_capture.container_stage import DockerReferenceStage


class TransportTests(unittest.TestCase):
    def bare_stage(self):
        stage=object.__new__(DockerReferenceStage)
        stage.verified_id='a'*64
        stage.child=Mock()
        stage.logs=[]
        return stage

    def test_failed_kill_rpc_does_not_manufacture_exit(self):
        stage=self.bare_stage()
        with patch('native.tests.reference_capture.container_stage.subprocess.run',
                   side_effect=subprocess.TimeoutExpired(['docker','kill'],1)),self.assertRaises(subprocess.TimeoutExpired):
            stage.terminate(stage.verified_id,'KILL',1)
        stage.child.wait.assert_not_called()

    def test_rpc_output_is_disk_backed_and_timeout_bounded(self):
        stage=self.bare_stage()
        with patch('native.tests.reference_capture.container_stage.subprocess.run') as run:
            run.return_value.returncode=0
            self.assertEqual(stage.rpc(['inspect',stage.verified_id],2),b'')
            self.assertEqual(run.call_args.kwargs['timeout'],2)
            self.assertNotIn('capture_output',run.call_args.kwargs)
            self.assertEqual(run.call_args.args[0][:2],['docker','inspect'])

    def test_missing_cgroup_or_unreaped_client_blocks_cleanup(self):
        for running in (False,True):
            stage=self.bare_stage()
            with tempfile.TemporaryDirectory() as folder:
                stage.cgroup=Path(folder)/'vanished'
                stage.manifest_path=Path(folder)/'manifest';stage.manifest_path.write_bytes(b'{}')
                stage.manifest_digest='0'*64;stage.final_oom=None
                stage.inspect=Mock(return_value={'State':{'Running':running,'Pid':0,'OOMKilled':False}})
                stage.child.poll.return_value=None
                stage.child.wait.side_effect=subprocess.TimeoutExpired(['docker','exec'],1)
                result=stage.observe(stage.verified_id,1)
                self.assertFalse(result.cleanup_known(stage.verified_id))
                self.assertIsNone(result.cgroup_oom_kill_delta)
                self.assertFalse(result.manifest_unchanged)

    def test_wrong_identity_cannot_signal(self):
        stage=self.bare_stage()
        with patch.object(stage,'rpc') as rpc,self.assertRaises(ValueError):stage.terminate('b'*64,'KILL',1)
        rpc.assert_not_called()


if __name__=='__main__':unittest.main()
