import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cuda_wait_monitor import cgroup_identity, task_counters, close_watchdog, publish_watchdog_error
from supervisor_cuda_wait import cleanup
import launch_cuda_wait as launch


class WaitMonitorTests(unittest.TestCase):
    def test_late_failure_and_error_write_failure_do_not_skip_teardown(self):
        monitor=Mock();monitor.close.side_effect=RuntimeError('late hot sample')
        child=Mock(pid=123)
        events=[]
        with patch('cuda_wait_monitor.Path.write_text',side_effect=OSError('disk')) as write, patch('supervisor_cuda_wait.os.killpg',side_effect=lambda *args:events.append('kill')), patch('supervisor_cuda_wait.time.monotonic',return_value=1):
            failure=close_watchdog(monitor)
            write.assert_not_called() # No evidence I/O before owner teardown.
            cleanup([child],30)
            events.append('restored')
            publish_watchdog_error('/unused',failure)
        self.assertEqual(events,['kill','restored'])
        child.wait.assert_called_once_with(timeout=29)
        self.assertIn('late hot sample',failure)


    def test_actual_cgroup_and_thread_identity(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);proc=root/'proc';cgroup=root/'cgroup';group=cgroup/'probe'
            task=proc/'12'/'task'/'13';task.mkdir(parents=True);group.mkdir(parents=True)
            (proc/'12'/'cgroup').write_text('0::/probe\n')
            for name,value in {'cpu.max':'max 100000','cpuset.cpus':'0,8','cpuset.cpus.effective':'0,8','memory.max':'8589934592','memory.swap.max':'0','cgroup.procs':'12\n'}.items():(group/name).write_text(value)
            fields=['0']*37;fields[11]='31';fields[12]='7';fields[19]='12345';fields[36]='8'
            stat='12 (named worker) '+' '.join(fields)
            (proc/'12'/'stat').write_text(stat);(task/'stat').write_text(stat)
            (task/'status').write_text('Cpus_allowed_list:\t0,8\n')
            path,metadata=cgroup_identity(12,proc,cgroup)
            self.assertEqual(metadata['cpu.max'],'max 100000')
            row=task_counters(path,proc)['processes'][0]
            self.assertEqual(row['start_ticks'],12345)
            self.assertEqual(row['threads'][0],dict(tid=13,name='named worker',start_ticks=12345,utime=31,stime=7,cpu=8,allowed='0,8'))


    def test_control_after_window_rejects_blocking_or_changed_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            row=dict(device=0,policy='control',commit='a'*40,gpu_uuid=launch.host.GPUS[0],flags={'before':4,'after':4,'after_window':4})
            (root/'rank-0.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'Inconclusive'):launch.verify(root,'control','a'*40)
            row['flags']={'before':0,'after':0,'after_window':4}
            (root/'rank-0.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'changed'):launch.verify(root,'control','a'*40)
