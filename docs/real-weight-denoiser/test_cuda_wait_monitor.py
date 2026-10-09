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

from cuda_wait_monitor import ThermalWatch, cgroup_identity, task_counters, close_watchdog, publish_watchdog_error
from nvml_sensor import NVMLReader
from supervisor_cuda_wait import cleanup
import launch_cuda_wait as launch


def sensor_result(stdout):
    rows={}
    for line in stdout.splitlines():
        uuid,used,free,temp=[v.strip() for v in line.split(',')]
        rows[uuid]=dict(used=int(used),free=int(free),temperature=int(temp))
    return dict(gpu=rows)


class WaitMonitorTests(unittest.TestCase):
    def test_slow_gpu_query_signals_main_independently(self):
        with tempfile.TemporaryDirectory() as root:
            monitor=ThermalWatch(root,['GPU-one'])
            with patch('cuda_wait_monitor.check_cpu',return_value=50) as cpu, patch('cuda_wait_monitor.NVMLReader.sample',side_effect=subprocess.TimeoutExpired('nvidia-smi',.35)) as query, patch('cuda_wait_monitor.os.kill') as kill:
                monitor._run()
            cpu.assert_called_once()
            self.assertGreater(query.call_args.kwargs['timeout'], .35)
            self.assertLessEqual(query.call_args.kwargs['timeout'], 1)
            self.assertIn('TimeoutExpired',monitor.error)
            self.assertEqual(kill.call_args.args[1],signal.SIGUSR1)

    def test_hot_cpu_signals_without_waiting_for_gpu_or_docker(self):
        with tempfile.TemporaryDirectory() as root:
            monitor=ThermalWatch(root,['GPU-one'])
            with patch('cuda_wait_monitor.check_cpu',side_effect=RuntimeError('hot')), patch('cuda_wait_monitor.NVMLReader.sample') as query,patch('cuda_wait_monitor.os.kill') as kill:
                monitor._run()
            query.assert_not_called();kill.assert_called_once()

    def test_evidence_failure_cannot_suppress_abort(self):
        with tempfile.TemporaryDirectory() as root:
            monitor=ThermalWatch(root,['GPU-one'])
            with patch('cuda_wait_monitor.check_cpu',side_effect=RuntimeError('hot')), patch('cuda_wait_monitor.Path.write_text',side_effect=OSError('disk')), patch('cuda_wait_monitor.os.kill') as kill:
                monitor._run()
            kill.assert_called_once()

    def test_late_failure_after_join_is_not_success(self):
        monitor=ThermalWatch('/unused',['GPU-one'])
        monitor.thread=Mock()
        monitor.thread.join.side_effect=lambda timeout:setattr(monitor,'error','late hot sample')
        monitor.thread.is_alive.return_value=False
        self.assertIn('late hot sample',close_watchdog(monitor))

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

    def run_sample(self, duration, *, cpu_duration=0, stdout='GPU-one, 0, 24000, 50\n', write_delay=0):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        monitor=ThermalWatch(directory.name,['GPU-one'])
        clock=[0.0]
        def cpu(*args): clock[0]+=cpu_duration; return 50
        def query(*args, **kwargs): clock[0]+=duration; return sensor_result(stdout)
        def wait(*args): monitor.stop_event.set()
        real_open=Path.open
        def opened(path,*args,**kwargs):
            clock[0]+=write_delay
            return real_open(path,*args,**kwargs)
        with patch('cuda_wait_monitor.time.monotonic',side_effect=lambda:clock[0]), patch('cuda_wait_monitor.check_cpu',side_effect=cpu), patch('cuda_wait_monitor.NVMLReader.sample',side_effect=query) as queried, patch('cuda_wait_monitor.Path.read_text',return_value='MemAvailable: 999999999 kB\n'), patch.object(monitor.stop_event,'wait',side_effect=wait), patch('cuda_wait_monitor.os.kill') as kill, patch('cuda_wait_monitor.Path.open',opened):
            monitor._run()
        return monitor,queried,kill,Path(directory.name)

    def test_slow_but_fresh_sample_records_actual_duration(self):
        monitor,query,kill,root=self.run_sample(.4,cpu_duration=.05)
        self.assertIsNone(monitor.error);kill.assert_not_called()
        self.assertAlmostEqual(query.call_args.kwargs['timeout'],.95)
        row=json.loads((root/'watchdog.jsonl').read_text())
        self.assertAlmostEqual(row['acquisition_seconds'],.45)
        self.assertAlmostEqual(row['query_seconds'],.4)
        self.assertAlmostEqual(row['sample_age_seconds'],.45)
        self.assertEqual(monitor.deadline,1)

    def test_exhausted_cpu_budget_never_launches_gpu_query(self):
        monitor,query,kill,_=self.run_sample(0,cpu_duration=1)
        query.assert_not_called();kill.assert_called_once()
        self.assertIn('stale before',monitor.error)

    def test_late_success_cannot_refresh_deadline(self):
        monitor,_,kill,_=self.run_sample(1)
        self.assertIsNone(monitor.last_valid);kill.assert_called_once()
        self.assertIn('stale after GPU',monitor.error)

    def test_evidence_delay_cannot_refresh_deadline(self):
        monitor,_,kill,_=self.run_sample(.2,write_delay=.9)
        self.assertIsNone(monitor.last_valid);kill.assert_called_once()
        self.assertIn('stale after evidence',monitor.error)

    def test_missing_malformed_hot_or_low_memory_gpu_fails_closed(self):
        for stdout in ('', 'malformed', 'GPU-one, 0, 24000, 90\n', 'GPU-one, 0, 255, 50\n'):
            with self.subTest(stdout=stdout):
                monitor,_,kill,_=self.run_sample(.1,stdout=stdout)
                self.assertIsNotNone(monitor.error);kill.assert_called_once()

    def test_check_rejects_stale_data_with_live_thread(self):
        monitor=ThermalWatch('/unused',['GPU-one']);monitor.deadline=1
        monitor.thread=Mock();monitor.thread.is_alive.return_value=True
        with patch('cuda_wait_monitor.time.monotonic',return_value=1),patch('cuda_wait_monitor.os.kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'stale'):monitor.check()
        kill.assert_called_once()

    def test_guard_aborts_blocked_acquisition_without_evidence_io(self):
        monitor=ThermalWatch('/unused',['GPU-one']);monitor.deadline=1
        monitor.stop_event=Mock();monitor.stop_event.wait.return_value=False
        with patch('cuda_wait_monitor.time.monotonic',return_value=1),patch('cuda_wait_monitor.os.kill') as kill,patch('cuda_wait_monitor.Path.write_text') as write:
            monitor._guard()
        kill.assert_called_once();write.assert_not_called()

    def test_blocked_cpu_read_is_interrupted_by_independent_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor=ThermalWatch(directory,['GPU-one'])
            monitor.freshness=.05;monitor.guard_interval=.005
            release=threading.Event();aborted=threading.Event()
            def blocked(*args): release.wait(2); raise RuntimeError('released')
            with patch('cuda_wait_monitor.check_cpu',side_effect=blocked),patch('cuda_wait_monitor.os.kill',side_effect=lambda *args:aborted.set()):
                try:
                    with self.assertRaisesRegex(RuntimeError,'stale'):monitor.start()
                    self.assertTrue(aborted.wait(.5))
                finally:
                    release.set()
                    with self.assertRaises(RuntimeError):monitor.close()
            self.assertFalse(monitor.thread.is_alive());self.assertFalse(monitor.guard.is_alive())

    def test_blocked_evidence_write_does_not_block_abort_or_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor=ThermalWatch(directory,['GPU-one'])
            monitor.freshness=.05;monitor.guard_interval=.005
            release=threading.Event();aborted=threading.Event()
            real_open=Path.open
            def blocked(path,*args,**kwargs):
                if path.name=='watchdog.jsonl': release.wait(2)
                return real_open(path,*args,**kwargs)
            with patch('cuda_wait_monitor.check_cpu',return_value=50),patch('cuda_wait_monitor.NVMLReader.sample',return_value=sensor_result('GPU-one, 0, 24000, 50\n')),patch('cuda_wait_monitor.Path.read_text',return_value='MemAvailable: 999999999 kB\n'),patch('cuda_wait_monitor.Path.open',blocked),patch('cuda_wait_monitor.os.kill',side_effect=lambda *args:aborted.set()):
                try:
                    with self.assertRaisesRegex(RuntimeError,'stale'):monitor.start()
                    self.assertTrue(aborted.wait(.5))
                    self.assertIsNone(monitor.last_valid)
                finally:
                    release.set()
                    with self.assertRaises(RuntimeError):monitor.close()
            self.assertFalse(monitor.thread.is_alive())

    def test_hung_nvml_helper_is_reaped_after_freshness_abort(self):
        with tempfile.TemporaryDirectory() as root:
            monitor=ThermalWatch(root,['GPU-one']);monitor.freshness=.1;monitor.guard_interval=.005
            monitor.reader=NVMLReader(['GPU-one'],command=[sys.executable,'-c','import time;time.sleep(60)'])
            real_kill=os.kill
            def signal_owner(pid,sig):
                if pid!=os.getpid():real_kill(pid,sig)
            try:
                with patch('cuda_wait_monitor.check_cpu',return_value=50),patch('cuda_wait_monitor.os.kill',side_effect=signal_owner):
                    with self.assertRaises(RuntimeError):monitor.start()
                    with self.assertRaises(RuntimeError):monitor.close()
            finally:monitor.reader.close()
            self.assertIsNotNone(monitor.reader.process.poll())
            self.assertFalse(monitor.thread.is_alive());self.assertFalse(monitor.guard.is_alive())

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
