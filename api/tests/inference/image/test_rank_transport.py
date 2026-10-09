import threading
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from api.inference.image.rank_session import RankBudget, RankSession
from api.inference.image.rank_transport import ProcessRanks, encode, rank_cpus
from api.inference.resources import ResourceManager, ResourceCancelled, ResourceBusy


class ProcessRankTests(unittest.TestCase):
    def setUp(self):
        self.memory = {0:0,1:0}
        self.budget = RankBudget(1000,(0,1),10,70)
        self.transport = ProcessRanks(Path('/unused'), self.budget,
            worker_module='api.tests.inference.image.rank_fixture', guard=lambda:None,
            memory_probe=lambda:dict(self.memory))
        self.resources = ResourceManager(2000,{0:100,1:100})
        self.session = RankSession(self.resources,self.transport,self.budget,enabled=True,
            operation_timeout=3,cleanup_timeout=.2)

    def tearDown(self):
        self.memory={0:0,1:0}
        self.session.close()

    def test_both_rank_startup_threads_share_one_two_cpu_budget(self):
        expected=rank_cpus(os.sched_getaffinity(0))
        replies=self.session.execute('a'*32,payload={'prompt':'ok'})
        union=set()
        for reply in replies:
            self.assertEqual(reply['main_affinity'],expected)
            self.assertEqual(reply['startup_affinity'],expected)
            union.update(reply['main_affinity']);union.update(reply['startup_affinity'])
        self.assertEqual(union,set(expected))
        self.assertLessEqual(len(union),2)

    def test_startup_failure_before_first_child_has_no_false_quarantine(self):
        self.transport.memory_probe=lambda: (_ for _ in ()).throw(OSError('probe unavailable'))
        with self.assertRaises(OSError):self.session.execute('a'*32)
        self.assertEqual(self.session.state,'closed')
        self.assertFalse(self.resources.snapshot()['reservations'])

    def test_real_children_reuse_park_restore_and_reap(self):
        self.session.execute('a'*32,payload={'prompt':'first'})
        children=list(self.transport.processes)
        self.session.park()
        self.assertEqual(self.session.state,'parked')
        self.session.execute('b'*32,payload={'prompt':'second'})
        self.assertEqual(children,self.transport.processes)
        self.session.close()
        self.assertTrue(all(p.poll() is not None for p in children))
        self.assertFalse(self.resources.snapshot()['reservations'])

    def test_peer_exit_and_stale_reply_fence_both_children(self):
        for marker in ('peer-exit','stale','oom'):
            with self.subTest(marker=marker):
                self.session.execute('a'*32,payload={'prompt':'ok'})
                children=list(self.transport.processes)
                with self.assertRaises((RuntimeError,EOFError,ValueError)):
                    self.session.execute('b'*32,payload={'prompt':marker})
                self.assertTrue(all(p.poll() is not None for p in children))
                self.assertFalse(self.resources.snapshot()['reservations'])

    def test_peer_stderr_is_bounded_and_survives_reap(self):
        self.session.execute('a' * 32, payload={'prompt': 'ok'})
        with self.assertLogs('api.inference.image.rank_transport', level='WARNING') as captured:
            with self.assertRaises((RuntimeError, EOFError)):
                self.session.execute('b' * 32, payload={'prompt': 'stderr-exit'})
        self.assertLessEqual(len(self.transport.stderr_tails[1]), 8192)
        self.assertTrue(self.transport.stderr_tails[1].endswith(b'fixture peer failure detail'))
        self.assertIn('fixture peer failure detail', '\n'.join(captured.output))
        self.assertEqual(self.transport.processes, [])
        self.assertFalse(self.resources.snapshot()['reservations'])

    def test_error_acknowledgement_is_retained(self):
        with self.assertLogs('api.inference.image.rank_transport', level='WARNING') as captured:
            with self.assertRaises(ValueError):
                self.session.execute('a' * 32, payload={'prompt': 'oom'})
        self.assertIn('synthetic allocation failure', '\n'.join(captured.output))
    def test_default_resource_admission_releases_rank_ownership(self):
        self.budget=RankBudget(128*1024**2,(0,1),10,70)
        self.transport=ProcessRanks(Path('/unused'),self.budget,
            worker_module='api.tests.inference.image.rank_fixture',
            memory_probe=lambda:dict(self.memory))
        self.resources=ResourceManager(256*1024**2,{0:100,1:100})
        self.session=RankSession(self.resources,self.transport,self.budget,enabled=True,
            operation_timeout=3,cleanup_timeout=.2)
        self.session.execute('a'*32,payload={'prompt':'ok'})
        self.transport._guard()
        children=list(self.transport.processes)
        self.session.close()
        self.assertTrue(all(p.poll() is not None for p in children))
        self.assertFalse(self.resources.snapshot()['reservations'])

    def test_cancelled_start_checks_before_memory_admission(self):
        cancel=threading.Event();cancel.set()
        with patch.object(self.transport,'memory_probe') as probe:
            with self.assertRaises(ResourceCancelled):
                self.transport.start('session',(0,1),time.monotonic()+1,cancel)
        probe.assert_not_called();self.assertIsNone(self.transport.directory)

    def test_default_guard_still_enforces_host_and_device_memory(self):
        with patch.object(self.transport,'_owned_fit',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'GPU envelope'):self.transport._guard()
        self.transport.processes=[SimpleNamespace(pid=999999)]
        try:
            with patch.object(self.transport,'_owned_fit',return_value=True),patch.object(Path,'read_text',return_value='VmRSS: 2 kB\n'):
                with self.assertRaisesRegex(RuntimeError,'host memory'):self.transport._guard()
        finally:self.transport.processes=[]

    def test_active_cancel_waits_for_both_reaps(self):
        self.session.execute('a'*32,payload={'prompt':'ok'})
        children=list(self.transport.processes);cancel=threading.Event()
        timer=threading.Timer(.1,cancel.set);timer.start()
        try:
            with self.assertRaises(ResourceCancelled):
                self.session.execute('b'*32,cancel,payload={'prompt':'wait'})
        finally:timer.join()
        self.assertTrue(all(p.poll() is not None for p in children))
        self.assertFalse(self.resources.snapshot()['reservations'])

    def test_unreleased_physical_memory_quarantines_even_after_reap(self):
        self.session.execute('a'*32,payload={'prompt':'ok'})
        children=list(self.transport.processes)
        self.memory[1]=100*1024**2
        with self.assertRaises(ResourceBusy):self.session.park()
        self.assertEqual(self.session.state,'quarantined')
        self.assertTrue(all(p.poll() is not None for p in children))
        self.assertEqual(len(self.resources.snapshot()['reservations']),3)
        with self.assertRaises(ResourceBusy):self.session.execute('b'*32)
        self.memory[1]=0;self.session.close(recover=True)
        self.assertFalse(self.resources.snapshot()['reservations'])

    def test_deadline_and_bounded_control_message(self):
        with self.assertRaises(ValueError):encode({'prompt':'x'*65536})
        self.session.operation_timeout=.1
        with self.assertRaises(TimeoutError):self.session.execute('a'*32,payload={'prompt':'wait'})
        self.assertFalse(self.resources.snapshot()['reservations'])


class RankCpuPlacementTests(unittest.TestCase):
    def test_spreads_two_cpus_across_last_level_domains_within_allowed_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for cpu,domain in ((0,'0-7,16-23'),(1,'0-7,16-23'),(8,'8-15,24-31'),(16,'0-7,16-23')):
                cache=root/f'cpu{cpu}'/'cache/index3';cache.mkdir(parents=True)
                (cache/'level').write_text('3');(cache/'shared_cpu_list').write_text(domain)
                lower=root/f'cpu{cpu}'/'cache/index0';lower.mkdir()
                (lower/'level').write_text('1');(lower/'shared_cpu_list').write_text(str(cpu))
            self.assertEqual(rank_cpus({16,8,1,0},root),[0,8])
            self.assertEqual(rank_cpus({0,1},root),[0,1])
            self.assertEqual(rank_cpus({8},root),[8])
            (root/'cpu0/cache/index3/level').write_text('unavailable')
            self.assertEqual(rank_cpus({0,1,8},root),[0,1])

    def test_missing_topology_never_widens_affinity_or_cpu_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertEqual(rank_cpus({7,3,9},root),[3,7])
            self.assertEqual(rank_cpus({9},root),[9])
            with self.assertRaises(RuntimeError):rank_cpus(set(),root)


class PhysicalOwnerTests(unittest.TestCase):
    def test_duplicate_canonical_devices_reject_before_child_launch(self):
        identity='e30b6419-2c6d-f550-61d6-16166a920dac'
        transport=ProcessRanks(Path('/unused'),RankBudget(1000,(0,1),10,70),guard=lambda:None)
        torch=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda d:
            SimpleNamespace(uuid=('GPU-' if d else '')+identity)))
        with patch('api.inference.image.rank_transport.subprocess.Popen') as spawn, \
                patch('api.inference.image.rank_transport.importlib.import_module',return_value=torch):
            with self.assertRaisesRegex(RuntimeError,'same physical GPU'):
                transport.start('session',(0,1),time.monotonic()+1,None)
        spawn.assert_not_called()
        self.assertEqual(transport.processes,[])
        self.assertIsNone(transport.directory)

    def test_torch_bare_uuid_matches_nvml_prefixed_identity(self):
        ids=['e30b6419-2c6d-f550-61d6-16166a920dac','2a2378dd-08c1-6f69-6317-a253d90e76b3']
        for prefix in ('','GPU-'):
            transport=ProcessRanks(Path('/unused'),RankBudget(1000,(0,1),10,70),guard=lambda:None)
            torch=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda d:SimpleNamespace(uuid=prefix+ids[d])))
            result=SimpleNamespace(stdout='\n'.join('GPU-'+identity+', 42' for identity in ids))
            with patch('api.inference.image.rank_transport.subprocess.run',return_value=result), \
                    patch('api.inference.image.rank_transport.importlib.import_module',return_value=torch):
                self.assertEqual(transport._device_usage(),{0:42*1024**2,1:42*1024**2})
            self.assertEqual(transport.uuids,{d:'GPU-'+value for d,value in enumerate(ids)})

    def test_park_checks_owned_pid_even_if_other_gpu_memory_disappears(self):
        budget=RankBudget(1000,(0,1),10,70)
        transport=ProcessRanks(Path('/unused'),budget,guard=lambda:None,memory_probe=lambda:{0:0,1:0})
        transport.uuids={0:'GPU-A',1:'GPU-B'}
        transport.baseline_processes={('GPU-A',10)}
        rows={('GPU-A',10):80,('GPU-A',20):9,('GPU-B',21):9}
        transport.process_probe=lambda:rows.copy()
        transport._confirm_owners()
        self.assertTrue(transport._owned_fit(10))
        del rows[('GPU-A',10)]
        rows[('GPU-A',20)]=11
        self.assertFalse(transport._owned_fit(10))
        self.assertFalse(transport._owned_fit(64*1024**2,stopped=True))
        rows.clear()
        self.assertTrue(transport._owned_fit(64*1024**2,stopped=True))

    def test_extra_gpu_process_during_startup_is_not_assumed_owned(self):
        transport=ProcessRanks(Path('/unused'),RankBudget(1000,(0,1),10,70),guard=lambda:None,memory_probe=lambda:{0:0,1:0})
        transport.uuids={0:'GPU-A',1:'GPU-B'}
        transport.process_probe=lambda:{('GPU-A',20):9,('GPU-A',22):9,('GPU-B',21):9}
        with self.assertRaises(RuntimeError):transport._confirm_owners()
