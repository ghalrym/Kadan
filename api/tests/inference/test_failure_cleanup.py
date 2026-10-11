import unittest
import weakref

from api.inference.failure_cleanup import clear_failure_frames


class FailureCleanupTests(unittest.TestCase):
    def test_chained_failure_releases_completed_frame_allocations(self):
        references = []
        class Allocation:
            pass
        def fail():
            allocation = Allocation()
            references.append(weakref.ref(allocation))
            raise ValueError('allocation failed')
        def wrap():
            try:
                fail()
            except ValueError as error:
                raise RuntimeError('wrapped') from error
        try:
            wrap()
        except RuntimeError as error:
            self.assertIsNotNone(references[0]())
            clear_failure_frames(error)
            self.assertIsNotNone(error.__cause__)
            self.assertIsNone(references[0]())
        else:
            self.fail('expected chained failure')
