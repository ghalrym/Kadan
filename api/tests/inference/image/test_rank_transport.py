import threading
import time
import unittest
from pathlib import Path

from api.inference.image.rank_session import RankBudget, RankSession
from api.inference.image.rank_transport import ProcessRanks, encode
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


class PhysicalOwnerTests(unittest.TestCase):
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
