import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cuda_wait_monitor import ThermalWatch, cgroup_identity, task_counters
import launch_cuda_wait as launch


class WaitMonitorTests(unittest.TestCase):
    def test_slow_gpu_query_signals_main_independently(self):
        with tempfile.TemporaryDirectory() as root:
            monitor=ThermalWatch(root,['GPU-one'])
            with patch('cuda_wait_monitor.check_cpu',return_value=50) as cpu, patch('cuda_wait_monitor.subprocess.run',side_effect=subprocess.TimeoutExpired('nvidia-smi',.35)) as query, patch('cuda_wait_monitor.os.kill') as kill:
                monitor._run()
            cpu.assert_called_once()
            self.assertEqual(query.call_args.kwargs['timeout'],.35)
            self.assertIn('TimeoutExpired',monitor.error)
            self.assertEqual(kill.call_args.args[1],signal.SIGUSR1)

    def test_hot_cpu_signals_without_waiting_for_gpu_or_docker(self):
        with tempfile.TemporaryDirectory() as root:
            monitor=ThermalWatch(root,['GPU-one'])
            with patch('cuda_wait_monitor.check_cpu',side_effect=RuntimeError('hot')), patch('cuda_wait_monitor.subprocess.run') as query,patch('cuda_wait_monitor.os.kill') as kill:
                monitor._run()
            query.assert_not_called();kill.assert_called_once()

    def test_missed_cadence_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            monitor=ThermalWatch(root,['GPU-one'])
            with patch('cuda_wait_monitor.time.monotonic',side_effect=[0,0,.2,1.2]), patch('cuda_wait_monitor.check_cpu',return_value=50), patch('cuda_wait_monitor.subprocess.run',return_value=SimpleNamespace(stdout='GPU-one, 0, 24000, 50\n')), patch('cuda_wait_monitor.Path.read_text',return_value='MemAvailable: 999999999 kB\n'), patch.object(monitor.stop_event,'wait'),patch('cuda_wait_monitor.os.kill') as kill:
                monitor._run()
            self.assertIn('cadence',monitor.error);kill.assert_called_once()

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


if __name__=='__main__':unittest.main()
