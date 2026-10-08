import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import launch_holdout
from capture_holdout import StepCounter, check_state
from holdout_contracts import BLOCK_KEYS, PROTOCOL, merge_context, safe_file, cleanup_successful_contexts, require_review, require_prior_step20, verdict
from holdout_packets import ContextWriter
from trace_binding import sha256


class HoldoutExtensionTests(unittest.TestCase):
    def context(self):
        values={key:torch.zeros(1) for key in BLOCK_KEYS}
        values.update(step_index=20,base_packet_sha256='base-sha')
        return values

    def test_step_counter_counts_prefill_without_off_by_one(self):
        for target in (20,39):
            counter=StepCounter(target)
            seen=[counter.observe('extract' if i==0 else 'cached',[float(40-i)]) for i in range(target+1)]
            self.assertEqual(seen,[False]*target+[True])
            self.assertEqual(len(counter.timesteps),target+1)
            with self.assertRaises(ValueError):counter.observe('cached',[0.])
        with self.assertRaises(ValueError):StepCounter(21)
        with self.assertRaises(ValueError):StepCounter(20).observe('cached',[1.])

    def test_context_merge_references_weights_and_rejects_duplicates(self):
        weights={'weight':torch.ones(2)};base={'state':weights,'hidden':torch.ones(1)}
        context=self.context();merged=merge_context(base,context,'base-sha',20)
        self.assertIs(merged['state'],weights)
        self.assertIs(merged['hidden'],context['hidden'])
        self.assertTrue(torch.equal(base['hidden'],torch.ones(1)))
        with self.assertRaises(ValueError):merge_context(base,dict(context,state=weights),'base-sha',20)
        with self.assertRaises(ValueError):merge_context(base,context,'wrong-sha',20)
        with self.assertRaises(ValueError):merge_context(base,context,'base-sha',39)

    def test_writer_preflights_budget_and_forbids_weight_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);writer=ContextWriter(root,0,limit=100,reserve=0)
            with self.assertRaises(RuntimeError):writer.write('block-00.context.pt',self.context())
            self.assertEqual({p.name for p in root.iterdir()},{'ownership.json'})
            with self.assertRaises(ValueError):writer.write('block-00.context.pt',dict(self.context(),state={}))
            with self.assertRaises(ValueError):safe_file(root,'../outside.pt')

    def fixture(self,root,passed=True):
        contexts=root/'contexts';contexts.mkdir();stage=root/'holdout-replay-evidence';stage.mkdir()
        rows=[]
        for name in ['block-00.context.pt','tail.context.pt']:
            (contexts/name).write_bytes(b'owned context')
            rows.append(dict(file=name,sha256=sha256(contexts/name)))
        manifest=dict(run_id='run',step_index=20,blocks=rows[:1],tail=rows[1])
        (contexts/'manifest.json').write_text(json.dumps(manifest))
        (contexts/'ownership.json').write_text(json.dumps(dict(run_id='run',files=[row['file'] for row in rows])))
        for rank in (0,1):
            state='passed' if passed or rank==0 else 'failed'
            (stage/f'verdict-rank-{rank}.json').write_text(json.dumps(verdict(20,state,sha256(contexts/'manifest.json'))))
        return contexts

    def test_cleanup_requires_both_passes_and_preserves_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'immutable-weights.pt';base.write_bytes(b'never delete')
            contexts=self.fixture(root)
            cleanup_successful_contexts(contexts,root)
            self.assertEqual(base.read_bytes(),b'never delete')
            self.assertEqual(list(contexts.iterdir()),[])
            self.assertTrue((root/'retained-context-manifest.json').exists())
            self.assertTrue((root/'context-cleanup.json').exists())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);contexts=self.fixture(root,passed=False)
            with self.assertRaises(ValueError):cleanup_successful_contexts(contexts,root)
            self.assertTrue((contexts/'block-00.context.pt').exists())
            self.assertFalse((root/'retained-context-manifest.json').exists())

    def test_cleanup_rejects_changed_or_unowned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);contexts=self.fixture(root)
            (contexts/'block-00.context.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):cleanup_successful_contexts(contexts,root)
            self.assertTrue((contexts/'tail.context.pt').exists())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);contexts=self.fixture(root)
            (contexts/'unrelated.txt').write_text('preserve')
            with self.assertRaises(ValueError):cleanup_successful_contexts(contexts,root)
            self.assertTrue((contexts/'unrelated.txt').exists())

    def test_default_plan_and_review_are_step_specific(self):
        with patch('sys.argv',['launch_holdout.py']),patch.object(launch_holdout,'run') as run,contextlib.redirect_stdout(io.StringIO()) as output:
            launch_holdout.main()
        run.assert_not_called();self.assertTrue(json.loads(output.getvalue())['shared_weights'])
        record=dict(protocol=PROTOCOL,source_commit='a'*40,step_index=20,decision='approved-for-bounded-execution',reviewer='reviewer',review_reference='review')
        require_review(record,'a'*40,20)
        with self.assertRaises(ValueError):require_review(record,'a'*40,39)
        row=verdict(20,'passed','digest');self.assertEqual(row['fp32_protocol'],'failed')
        self.assertFalse(row['production_activation']);self.assertEqual(row['decoded_image'],'not-run')

    def test_prior_step_requires_same_manifest_restoration_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);contexts=self.fixture(root)
            cleanup_successful_contexts(contexts,root)
            (root/'pause-result.json').write_text(json.dumps(dict(restored=True)))
            require_prior_step20(root)
            row=root/'holdout-replay-evidence'/'verdict-rank-1.json'
            value=json.loads(row.read_text());value['context_manifest_sha256']='different'
            row.write_text(json.dumps(value))
            with self.assertRaises(ValueError):require_prior_step20(root)

    def test_changed_weights_are_rejected(self):
        check_state({'a':torch.ones(2)},{'a':torch.ones(2)})
        with self.assertRaises(ValueError):check_state({'a':torch.zeros(2)},{'a':torch.ones(2)})


if __name__=='__main__':unittest.main()
