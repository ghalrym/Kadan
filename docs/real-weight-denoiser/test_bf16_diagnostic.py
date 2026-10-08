import contextlib
import io
import json
import signal
import subprocess
import unittest
from unittest.mock import patch, Mock, mock_open

import torch

import launch_bf16
import supervisor_bf16
from bf16_contracts import PROTOCOL, require_ci, require_review, require_pass, verdict, timing_admission, aggregate
from bf16_numerics import inspect_values, expected_head_ownership, advance_pair


class BF16DiagnosticTests(unittest.TestCase):
    def test_default_launcher_is_read_only_and_keeps_fp32_failed(self):
        with patch('sys.argv',['launch_bf16.py']),patch.object(launch_bf16,'run') as run,contextlib.redirect_stdout(io.StringIO()) as output:
            launch_bf16.main()
        run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['prior_fp32'],'failed')

    def test_independent_review_and_exact_head_ci_required(self):
        commit='a'*40
        with self.assertRaises(ValueError):require_review({},commit)
        record=dict(protocol=PROTOCOL,source_commit=commit,decision='approved-for-bounded-execution',reviewer='independent-review',review_reference='review-message')
        require_review(record,commit)
        with self.assertRaises(ValueError):require_review(record,'b'*40)
        runs=[dict(name=n,headSha=commit,event='push',status='completed',conclusion='success') for n in ('API tests','Native worker CPU tests')]
        require_ci(runs,commit)
        runs[0]['conclusion']='failure'
        with self.assertRaises(ValueError):require_ci(runs,commit)
        with self.assertRaises(ValueError):require_ci([],commit)

    def test_rank_failures_and_nonfinite_cannot_be_averaged_away(self):
        good=inspect_values(torch.tensor([0.,1.]),torch.tensor([0.,1.]))
        bad=inspect_values(torch.tensor([.021,1.]),torch.tensor([0.,1.]))
        self.assertEqual((bad['atol'],bad['rtol']),(.02,.02))
        with self.assertRaises(AssertionError):require_pass([good,bad])
        with self.assertRaises(AssertionError):require_pass([good,inspect_values(torch.tensor([float('nan')]),torch.zeros(1))])
        self.assertEqual(require_pass([good,None])['count'],2)
        self.assertEqual(bad['max_normalized_point']['flat_index'],0)

    def test_explicit_head_ownership_detects_canceling_permutations(self):
        full=torch.arange(1*6*4*2).reshape(1,6,4,2)
        expected=expected_head_ownership(full,1,2)
        wrong=expected.flip(1)
        self.assertTrue(torch.equal(wrong.flip(1),expected)) # Round trip can hide a bug.
        self.assertFalse(torch.equal(wrong,full[:,:,2:4]))
        self.assertTrue(torch.equal(expected,full[:,:,2:4]))

    def test_paired_chains_do_not_reset_or_share_state(self):
        reference=2;parallel=3
        for _ in range(4):reference,parallel=advance_pair(reference,parallel,lambda x:x*2,lambda x:x+5)
        self.assertEqual((reference,parallel),(32,23))
        self.assertNotEqual(parallel,reference)

    def test_passed_first_slice_never_clears_old_or_pending_gates(self):
        row=verdict('passed','passed-cached-core-only')
        self.assertEqual(row['fp32_protocol'],'failed')
        self.assertEqual(row['overall_protocol'],'incomplete')
        self.assertEqual(row['later_steps'],{'20':'not-run','39':'not-run'})

    def test_global_histogram_and_worst_point_are_not_rank_averages(self):
        a=inspect_values(torch.zeros(9),torch.zeros(9))
        b=inspect_values(torch.tensor([.03]),torch.zeros(1))
        a['rank']=0;b['rank']=1
        b['max_normalized_point']['global_coordinate']=[0,8192,0]
        result=aggregate([a,b])
        self.assertEqual(result['finite_error_histogram']['counts'],
            [x+y for x,y in zip(a['error_histogram_counts'],b['error_histogram_counts'])])
        self.assertEqual(result['absolute_error_quantile_bins']['0.5'],[0.,1e-5])
        self.assertEqual(result['absolute_error_quantile_bins']['0.99'],[.02,.05])
        self.assertEqual(result['max_normalized_point']['global_coordinate'],[0,8192,0])
        self.assertEqual(result['max_normalized_point']['rank'],1)
        self.assertEqual(result['max_normalized_point']['reference'],0.)
        self.assertAlmostEqual(result['max_normalized_point']['bound'],.02)
        self.assertAlmostEqual(result['rmse'],.03/(10**.5),places=8)

    def test_global_nonfinite_metrics_are_unavailable_not_zero(self):
        a=inspect_values(torch.tensor([0.,float('nan')]),torch.zeros(2))
        b=inspect_values(torch.tensor([.03]),torch.zeros(1))
        result=aggregate([a,b])
        self.assertEqual(result['nonfinite'],1)
        self.assertEqual(result['finite_count'],2)
        for key in ('rmse','relative_l2','max_abs_error','max_normalized_tolerance_ratio',
                    'max_normalized_point','absolute_error_quantile_bins'):
            self.assertIsNone(result[key],key)
        self.assertEqual(sum(result['finite_error_histogram']['counts']),2)
        self.assertEqual(result['finite_error_histogram']['scope'],'finite-elements-only')
        with self.assertRaises(AssertionError):require_pass([a,b])

    def test_timing_admission_uses_minimum_asymmetric_budget(self):
        for budgets in ([119.999,150.],[150.,119.999],[0.,900.],[-1.,900.]):
            decision=timing_admission(budgets)
            self.assertFalse(decision['admitted'])
            self.assertEqual(decision['minimum_remaining_seconds'],min(budgets))
        for budgets in ([120.,150.],[150.,120.]):
            self.assertTrue(timing_admission(budgets)['admitted'])
        with self.assertRaises(ValueError):timing_admission([120.])
        with self.assertRaises(ValueError):timing_admission([float('nan'),150.])

    def test_supervisor_timeout_reaps_only_owned_process_group(self):
        child=Mock(pid=12345)
        child.wait.side_effect=[subprocess.TimeoutExpired('owned',1),0]
        child.poll.return_value=0
        with patch('sys.argv',['supervisor_bf16.py','bf16-replay']),patch.dict('os.environ',{'KADAN_STAGE_SECONDS':'1'}), \
                patch('builtins.open',mock_open()),patch.object(supervisor_bf16.fcntl,'flock'), \
                patch.object(supervisor_bf16.subprocess,'Popen',return_value=child), \
                patch.object(supervisor_bf16.os,'killpg') as kill,patch.object(supervisor_bf16.signal,'signal'), \
                patch.object(supervisor_bf16.Path,'write_text'):
            with self.assertRaises(subprocess.TimeoutExpired):supervisor_bf16.main()
        self.assertEqual([call.args for call in kill.call_args_list],[(12345,signal.SIGTERM),(12345,signal.SIGKILL)])

    def test_new_launcher_keeps_absolute_recovery_reserve(self):
        budget=launch_bf16.PauseBudget(0.)
        with patch.object(launch_bf16.time,'monotonic',return_value=1100.):
            self.assertEqual(budget.stage_seconds(),370)
        self.assertEqual(budget.end-budget.measurement_end,600)
        with patch.object(launch_bf16,'COMMAND_DEADLINE',10.),patch.object(launch_bf16.time,'monotonic',return_value=9.),patch.object(launch_bf16.subprocess,'run') as run:
            launch_bf16.run('docker','start','exact-id',timeout=45)
            self.assertEqual(run.call_args.kwargs['timeout'],1.)


if __name__=='__main__':unittest.main()
