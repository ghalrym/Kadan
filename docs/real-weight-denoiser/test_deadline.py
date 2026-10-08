"""Absolute pause budget includes stop, stage transitions, cleanup and recovery."""
import unittest
from unittest.mock import patch

import launch


class PauseDeadlineTests(unittest.TestCase):
    def test_second_stage_is_shortened_by_elapsed_stop_and_cleanup(self):
        budget=launch.PauseBudget(100.)
        with patch.object(launch.time,'monotonic',return_value=145.):
            self.assertEqual(budget.stage_seconds(),900)
        # Stop45s, first stage900s, transition60s: no fresh900s allocation.
        with patch.object(launch.time,'monotonic',return_value=1105.):
            self.assertEqual(budget.stage_seconds(),465)
        self.assertEqual(budget.end-budget.measurement_end,600)

    def test_expired_measurement_cannot_start_another_stage(self):
        budget=launch.PauseBudget(0.)
        with patch.object(launch.time,'monotonic',return_value=1470.):
            with self.assertRaises(TimeoutError):budget.stage_seconds()

    def test_commands_are_clipped_to_absolute_deadline(self):
        with patch.object(launch,'COMMAND_DEADLINE',100.), patch.object(launch.time,'monotonic',return_value=97.), \
                patch.object(launch.subprocess,'run') as run:
            launch.run('docker','start','original-id',timeout=45)
            self.assertEqual(run.call_args.kwargs['timeout'],3.)
        with patch.object(launch,'COMMAND_DEADLINE',100.), patch.object(launch.time,'monotonic',return_value=100.), \
                patch.object(launch.subprocess,'run') as run:
            with self.assertRaises(TimeoutError):launch.run('docker','start','original-id')
            run.assert_not_called()

    def test_readiness_cannot_add_time_past_overall_deadline(self):
        budget=launch.PauseBudget(0.)
        with patch.object(launch.time,'monotonic',return_value=2099.):
            self.assertEqual(launch.remaining(budget.end,3),1.)
        with patch.object(launch.time,'monotonic',return_value=2100.):
            with self.assertRaises(TimeoutError):launch.remaining(budget.end,3)


if __name__=='__main__':unittest.main()
