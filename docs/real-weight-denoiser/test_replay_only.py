import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch_replay_only
import replay_bounded
from replay_only_contracts import require_review, PROTOCOL, CONTEXT_SHA, verify_retained


class ReplayOnlyTests(unittest.TestCase):
    def test_default_is_read_only(self):
        with patch('sys.argv',['launch_replay_only.py']),patch.object(launch_replay_only,'run') as run,contextlib.redirect_stdout(io.StringIO()) as output:
            launch_replay_only.main()
        run.assert_not_called();plan=json.loads(output.getvalue())
        self.assertEqual(plan['cases'],['replay-only-step-39']);self.assertEqual(plan['cpu_quota'],2)

    def test_review_binds_source_and_retained_context(self):
        record=dict(protocol=PROTOCOL,source_commit='commit',context_manifest_sha256=CONTEXT_SHA,decision='approved-for-bounded-execution',reviewer='r',review_reference='ref')
        require_review(record,'commit')
        for key,value in [('source_commit','wrong'),('context_manifest_sha256','wrong'),('protocol','wrong')]:
            with self.assertRaises(ValueError):require_review(dict(record,**{key:value}),'commit')




    def test_retained_wrong_hash_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'contexts').mkdir();(root/'contexts/manifest.json').write_text('{}')
            with patch('replay_only_contracts.require_prior_step20'):
                with self.assertRaises(ValueError):verify_retained(root,{},root)

    def test_unbounded_cgroup_is_rejected(self):
        env={'KADAN_REVIEWED_COMMIT':'new','KADAN_CAPTURE_COMMIT':'original','LOCAL_RANK':'0'}
        with patch.dict(replay_bounded.os.environ,env),patch.object(replay_bounded,'Path') as path, \
                patch.object(replay_bounded.torch,'set_num_threads'),patch.object(replay_bounded.torch,'set_num_interop_threads'), \
                patch.object(replay_bounded.torch,'get_num_threads',return_value=1),patch.object(replay_bounded.torch,'get_num_interop_threads',return_value=1):
            for quota in ('max 100000','400000 100000'):
                path.return_value.read_text.return_value=quota
                with self.assertRaises(RuntimeError):replay_bounded.configure_threads()
            path.return_value.read_text.return_value='200000 100000'
            self.assertEqual(replay_bounded.configure_threads()['interop'],1)

    def test_deadline_still_reserves_recovery(self):
        budget=launch_replay_only.PauseBudget(0)
        with patch.object(launch_replay_only.time,'monotonic',return_value=1100):self.assertEqual(budget.stage_seconds(),370)
        self.assertEqual(budget.end-budget.measurement_end,600)


if __name__=='__main__':unittest.main()
