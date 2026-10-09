import threading
import unittest

from api.inference.image.rank_session import RankBudget, RankSession
from api.inference.resources import ResourceManager, ResourceBusy, ResourceCancelled, ResourceExhausted


class FakeRanks:
    def __init__(self):
        self.starts = 0
        self.stops = 0
        self.commands = []
        self.confirmed = True
        self.mutate = None
        self.callback = None

    def start(self, session, devices, deadline, cancel):
        self.starts += 1
        self.devices = devices

    def exchange(self, command, deadline, cancel):
        self.commands.append((command.copy(), deadline))
        if self.callback:
            self.callback(command)
        replies = [dict(command, rank=rank, device=self.devices[rank], status='ok',
            resident_bytes=0 if command['operation'] == 'park' else 20) for rank in (0, 1)]
        return self.mutate(command, replies) if self.mutate else replies

    def stop(self, deadline):
        self.stops += 1
        return self.confirmed


class RankSessionTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceManager(1000, {0: 100, 1: 100})
        self.transport = FakeRanks()
        self.controller = RankSession(self.resources, self.transport, RankBudget(200, (0, 1), 10, 70), enabled=True)

    def reservations(self):
        return self.resources.snapshot()['reservations']

    def test_cancel_during_park_fences_before_releasing_either_device(self):
        self.controller.execute('a'*32)
        cancel=threading.Event()
        self.transport.callback=lambda command: cancel.set() if command['operation']=='park' else None
        with self.assertRaises(ResourceCancelled):self.controller.park(cancel)
        self.assertEqual(self.transport.stops,1)
        self.assertEqual(self.controller.state,'closed')
        self.assertFalse(self.reservations())

    def test_disabled_has_no_allocation_or_child(self):
        self.controller.enabled = False
        with self.assertRaises(ResourceBusy):
            self.controller.execute('a' * 32)
        self.assertFalse(self.reservations())
        self.assertEqual(self.transport.starts, 0)

    def test_consecutive_jobs_reuse_both_ranks_and_reservations(self):
        self.controller.execute('a' * 32)
        self.controller.execute('b' * 32)
        self.assertEqual(self.transport.starts, 1)
        self.assertEqual([row[0]['operation'] for row in self.transport.commands], ['ready', 'execute', 'execute'])
        self.assertEqual(self.reservations()[self.controller.owner + ':execution']['device_bytes'], {0: 70, 1: 70})
        with self.assertRaises(ValueError):
            self.controller.execute('b' * 32)
        self.controller.close()
        self.assertFalse(self.reservations())

    def test_park_releases_both_execution_budgets_retains_contexts_and_ram(self):
        self.controller.execute('a' * 32)
        self.controller.park()
        self.assertEqual(self.controller.state, 'parked')
        self.assertNotIn(self.controller.owner + ':execution', self.reservations())
        self.assertEqual(self.reservations()[self.controller.owner + ':context']['device_bytes'], {0: 10, 1: 10})
        text = self.resources.reserve('text', 'llm', device_bytes={0: 90, 1: 90})
        text.release()
        self.controller.execute('b' * 32)
        self.assertEqual(self.transport.starts, 1)
        self.assertEqual([c['operation'] for c, _ in self.transport.commands], ['ready', 'execute', 'park', 'restore', 'execute'])
        self.controller.close()

    def test_one_card_short_never_starts_or_leaks_partial_admission(self):
        held = self.resources.reserve('other', 'llm', device_bytes={1: 40})
        with self.assertRaises(ResourceExhausted):
            self.controller.execute('a' * 32)
        self.assertEqual(self.transport.starts, 0)
        self.assertEqual(set(self.reservations()), {'other'})
        held.release()

    def test_cancel_waits_for_reap_before_releasing_accounting(self):
        cancel = threading.Event()
        def callback(command):
            if command['operation'] == 'execute':
                self.assertEqual(len(self.reservations()), 3)
                cancel.set()
        self.transport.callback = callback
        with self.assertRaises(ResourceCancelled):
            self.controller.execute('a' * 32, cancel)
        self.assertEqual(self.transport.stops, 1)
        self.assertFalse(self.reservations())

    def test_stale_or_missing_peer_fails_and_closes_whole_session(self):
        for mutation in (lambda c, r: r[:1], lambda c, r: [dict(r[0], sequence=0), r[1]],
                         lambda c, r: [r[0], dict(r[1], rank=0)]):
            with self.subTest(mutation=mutation):
                self.transport.mutate = mutation
                with self.assertRaises(ValueError):
                    self.controller.execute('a' * 32)
                self.assertFalse(self.reservations())
                self.assertEqual(self.controller.state, 'closed')

    def test_uncertain_cleanup_quarantines_all_bytes_until_explicit_recovery(self):
        self.controller.execute('a' * 32)
        self.transport.confirmed = False
        with self.assertRaises(ResourceBusy):
            self.controller.close()
        self.assertEqual(self.controller.state, 'quarantined')
        self.assertEqual(len(self.reservations()), 3)
        with self.assertRaises(ResourceBusy):
            self.controller.execute('b' * 32)
        self.transport.confirmed = True
        self.controller.close()
        self.assertFalse(self.reservations())

    def test_park_requires_zero_model_residency_from_both_ranks(self):
        self.controller.execute('a' * 32)
        self.transport.mutate = lambda c, r: [r[0], dict(r[1], resident_bytes=1)]
        with self.assertRaises(ValueError):
            self.controller.park()
        self.assertEqual(self.transport.stops, 1)
        self.assertFalse(self.reservations())

    def test_one_absolute_deadline_covers_load_and_execute(self):
        now = [10.]
        self.controller.clock = lambda: now[0]
        self.controller.operation_timeout = 5
        def callback(command):
            now[0] += 3
        self.transport.callback = callback
        with self.assertRaises(TimeoutError):
            self.controller.execute('a' * 32)
        self.assertEqual([deadline for _, deadline in self.transport.commands], [15., 15.])
        self.assertFalse(self.reservations())

    def test_no_second_queue_or_concurrent_bypass(self):
        entered, release = threading.Event(), threading.Event()
        def hold():
            with self.controller.gate:
                entered.set()
                release.wait(2)
        thread = threading.Thread(target=hold)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            with self.assertRaises(ResourceBusy):
                self.controller.execute('a' * 32)
            self.assertEqual(self.transport.starts, 0)
        finally:
            release.set()
            thread.join(2)

    def test_budget_and_time_bounds_are_explicit(self):
        with self.assertRaises(ValueError):
            RankBudget(100, (0, 1, 0), 10, 20)
        with self.assertRaises(ValueError):
            RankSession(self.resources, self.transport, RankBudget(100, (0, 1), 10, 20), operation_timeout=float('inf'))


if __name__ == '__main__':
    unittest.main()
