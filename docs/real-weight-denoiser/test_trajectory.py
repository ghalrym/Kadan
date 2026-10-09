import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, mock_open

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor

import launch_trajectory
import supervisor_trajectory
from trajectory_adapter import SequenceParallelQwenTransformer, validate_transformer_output
from trajectory_contracts import StepOrder, storage_plan, require_review, PROTOCOL, SETTINGS, CRITERIA, verify_run, require_predecessors
from trace_binding import sha256
from trajectory_io import measure, normalized_rgba, ArtifactWriter, tensor_identity


class TrajectoryTests(unittest.TestCase):
    def test_complete_order_requires_own_prediction_per_step(self):
        order=StepOrder()
        with self.assertRaises(ValueError):order.scheduler()
        for i in range(40):
            order.prediction(i,'extract' if i==0 else 'cached')
            with self.assertRaises(ValueError):order.prediction(i,'cached')
            self.assertEqual(order.scheduler(),i)
        order.complete()
        with self.assertRaises(ValueError):order.prediction(40,'cached')

    def test_incomplete_and_reset_paths_rejected(self):
        order=StepOrder();order.prediction(0,'extract');order.scheduler()
        with self.assertRaises(ValueError):order.complete()
        with self.assertRaises(ValueError):order.prediction(0,'extract')
        with self.assertRaises(ValueError):order.prediction(1,'extract')

    def test_pixel_limits_and_unsigned_difference(self):
        reference=torch.zeros(4,dtype=torch.uint8)
        self.assertTrue(measure(torch.tensor([2,0,0,0],dtype=torch.uint8),reference,'pixel')['passed'])
        self.assertFalse(measure(torch.tensor([3,0,0,0],dtype=torch.uint8),reference,'pixel')['passed'])
        self.assertFalse(measure(torch.tensor([2,2,0,0],dtype=torch.uint8),reference,'pixel')['passed'])
        self.assertEqual(measure(torch.zeros(1,dtype=torch.uint8),torch.tensor([255],dtype=torch.uint8),'pixel')['max_abs_error'],255)

    def test_float_limits_and_nonfinite_before_clipping(self):
        ref=torch.zeros(4)
        self.assertTrue(measure(torch.tensor([1/255,0,0,0]),ref,'float')['passed'])
        self.assertFalse(measure(torch.full((4,),1/255),ref,'float')['passed'])
        post=Mock()
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(AssertionError):normalized_rgba(torch.tensor([value]),post)
        post.assert_not_called()
        post.return_value=np.zeros((1,2,2,4),dtype=np.float32)
        normalized_rgba(torch.ones(1),post,width=2,height=2);self.assertEqual(post.call_args.kwargs,{'output_type':'np'})

    def test_pinned_processor_preserves_rgba_and_alpha_rounding(self):
        processor=VaeImageProcessor()
        raw=torch.tensor([-.5,0.,.5,1.],dtype=torch.bfloat16).reshape(1,4,1,1)
        rgba=normalized_rgba(raw,processor.postprocess,width=1,height=1)
        np.testing.assert_array_equal(rgba,np.array([[[[.25,.5,.75,1.]]]],dtype=np.float32))
        image=processor.numpy_to_pil(rgba)[0]
        self.assertEqual(image.mode,'RGBA')
        self.assertEqual(image.getpixel((0,0)),(64,128,191,255))
        reference=torch.from_numpy(rgba.copy())
        changed=reference.clone();changed[...,3]-=3/255
        self.assertFalse(measure(changed,reference,'float')['passed'])
        pixels=torch.tensor([[[64,128,191,255]]],dtype=torch.uint8)
        altered=pixels.clone();altered[...,3]=252
        self.assertFalse(measure(altered,pixels,'pixel')['passed'])
        altered[...,3]=254
        self.assertFalse(measure(altered,pixels,'pixel')['passed'])
        changed=reference.clone();changed[...,0:3]-=.6/255
        self.assertFalse(measure(changed,reference,'float')['passed'])

    def test_rgb_or_wrong_dtype_contract_reports_actual_shape(self):
        for value in (np.zeros((1,2,2,3),dtype=np.float32),np.zeros((1,2,2,4),dtype=np.float64)):
            with self.assertRaisesRegex(ValueError,'Expected float32 RGBA'):
                normalized_rgba(torch.ones(1),Mock(return_value=value),width=2,height=2)

    def test_latent_gate_not_replaced_by_image_gate(self):
        self.assertFalse(measure(torch.tensor([.1]),torch.zeros(1),'latent')['passed'])
        self.assertTrue(measure(torch.tensor([.01]),torch.zeros(1),'latent')['passed'])
        with self.assertRaises(ValueError):measure(torch.zeros(1),torch.zeros(1,dtype=torch.bfloat16),'latent')

    def test_streamed_packet_owns_storage_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=ArtifactWriter(directory);parent=torch.zeros(1000);value=parent[:2]
            writer.tensor('initial.pt',value);parent.fill_(9)
            actual=torch.load(Path(directory)/'initial.pt',weights_only=True)
            self.assertEqual(actual.tolist(),[0.,0.]);self.assertEqual(actual.untyped_storage().nbytes(),8)
            with self.assertRaises(ValueError):writer.tensor('initial.pt',value)
            with self.assertRaises(ValueError):writer.tensor('../outside.pt',value)

    def test_storage_plan_counts_all_three_runs_and_failure_reserve(self):
        plan=storage_plan(31748000000)
        self.assertLess(plan['projected_total'],plan['limit']);self.assertEqual(plan['runs'],3)
        with self.assertRaises(ValueError):storage_plan(plan['limit'])

    def test_review_requires_case_criteria_and_settings(self):
        record=dict(protocol=PROTOCOL,source_commit='commit',case='reference',criteria=CRITERIA,settings=SETTINGS,decision='approved-for-bounded-execution',reviewer='r',review_reference='ref')
        require_review(record,'commit','reference')
        with self.assertRaises(ValueError):require_review(record,'commit','candidate')
        with self.assertRaises(ValueError):require_review(dict(record,criteria=dict(CRITERIA,pixel_max=3)),'commit','reference')

    def test_prior_incomplete_repeat_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest=dict(protocol=PROTOCOL,case='repeat',source_commit='commit',settings=SETTINGS,criteria=CRITERIA,status='incomplete',completed_steps=40)
            (Path(directory)/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_run(directory,'repeat','commit')

    def test_default_launcher_never_stops_api(self):
        with patch('sys.argv',['launch_trajectory.py']),patch.object(launch_trajectory,'run') as run,contextlib.redirect_stdout(io.StringIO()) as output:
            launch_trajectory.main()
        run.assert_not_called();self.assertEqual(json.loads(output.getvalue())['cases'],['reference','repeat','candidate'])

    def test_cached_adapter_carries_local_output_and_independent_prefix(self):
        blocks=[SimpleNamespace(forward=Mock(return_value='prefill')) for _ in range(2)]
        norm=SimpleNamespace(forward=lambda hidden,temb,mask:hidden)
        transformer=SimpleNamespace(transformer_blocks=blocks,norm_out=norm)
        adapter=SequenceParallelQwenTransformer(transformer,0,None);adapter.install()
        self.assertEqual(blocks[0].forward(), 'prefill')
        adapter.mode='cached';adapter.step=1
        # Zero-stride source avoids allocating a full 256-MiB fixture.
        full=torch.zeros(1).expand(1,16384,4096);rope=torch.zeros(1,dtype=torch.complex64).expand(16384,64)
        prefix=(torch.zeros(1,1,32,128),torch.zeros(1,1,32,128));cache=SimpleNamespace(get=lambda:prefix)
        args=dict(kv_cache_mode='cached',modulation=torch.zeros(2,16384),rotary_emb=rope,target_token_mask=torch.ones(16384,dtype=torch.bool),layer_cache=cache,attention_mask=None)
        seen=[]
        def step(block,hidden,*args,**kwargs):seen.append(hidden.shape);return hidden
        with patch('trajectory_adapter.cached_block',side_effect=step):
            local=blocks[0].forward(hidden_states=full,**args)
            result=blocks[1].forward(hidden_states=local,**args)
        self.assertIs(result,local);self.assertEqual(seen,[(1,8192,4096)]*2)
        self.assertNotEqual(adapter.prefix[0][0].data_ptr(),prefix[0].data_ptr());adapter.close()

    def fixture_run(self,root,case,reference=None):
        stage=root/'trajectory-evidence';stage.mkdir(parents=True)
        names=['initial.pt','float-rgba.pt','pixels.pt','output.png','input-identity.json']
        names += [f'{kind}-{step:02}.pt' for kind in ('prediction','latent') for step in range(40)]
        rows=[]
        for name in names:
            (stage/name).write_bytes(b'fixture');rows.append(dict(file=name,bytes=7,sha256=sha256(stage/name)))
        manifest=dict(protocol=PROTOCOL,case=case,source_commit='commit',settings=SETTINGS,criteria=CRITERIA,status='passed',completed_steps=40,raw_vae_finite=True,artifacts=rows,
            reference_manifest_sha256=None if reference is None else sha256(reference/'trajectory-evidence/manifest.json'))
        if case!='reference':
            report=stage/'comparisons-rank-0.jsonl';report.write_text(''.join(json.dumps(dict(passed=True))+'\n' for _ in range(83)))
            manifest.update(comparisons_count=83,comparison_reports=[dict(file=report.name,sha256=sha256(report))])
        (stage/'manifest.json').write_text(json.dumps(manifest));(root/'pause-result.json').write_text(json.dumps(dict(restored=True)))

    def test_candidate_requires_complete_repeat_for_exact_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'reference';repeat=root/'repeat';self.fixture_run(reference,'reference')
            with self.assertRaises(ValueError):require_predecessors('candidate','commit',reference)
            self.fixture_run(repeat,'repeat',reference);require_predecessors('candidate','commit',reference,repeat)
            (repeat/'trajectory-evidence/prediction-19.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):require_predecessors('candidate','commit',reference,repeat)

    def test_target_gather_restores_rank_row_order(self):
        adapter=SequenceParallelQwenTransformer(None,0,None);adapter.mode='cached'
        output=torch.ones(1,8192,64)
        def gather(parts,value):parts[0].copy_(value);parts[1].fill_(2)
        with patch('trajectory_adapter.dist.all_gather',side_effect=gather):full=adapter.gather(output)
        self.assertTrue(torch.equal(full[:,:8192],output));self.assertTrue(bool((full[:,8192:]==2).all()))

    def test_cancel_reaps_owned_group_before_unlock(self):
        child=Mock(pid=12345);child.wait.side_effect=[InterruptedError('cancel'),0];child.poll.return_value=0
        events=[]
        with patch.dict(supervisor_trajectory.os.environ,{'KADAN_STAGE_SECONDS':'900','KADAN_TRAJECTORY_CASE':'candidate'}), \
                patch('sys.argv',['supervisor_trajectory.py','trajectory']),patch('builtins.open',mock_open()), \
                patch.object(supervisor_trajectory.fcntl,'flock'),patch.object(supervisor_trajectory.signal,'signal'), \
                patch.object(supervisor_trajectory.subprocess,'Popen',return_value=child), \
                patch.object(supervisor_trajectory.os,'killpg',side_effect=lambda pid,sig:events.append((pid,sig))), \
                patch.object(supervisor_trajectory,'Path'):
            with self.assertRaises(InterruptedError):supervisor_trajectory.main()
        self.assertEqual(child.wait.call_count,2)
        self.assertEqual([event[0] for event in events],[12345,12345])
        self.assertEqual(events[-1][1],supervisor_trajectory.signal.SIGKILL)

    def test_extract_prefix_is_preserved_until_pipeline_target_slice(self):
        raw=torch.arange(16389).view(1,16389,1).expand(1,16389,64)
        self.assertIs(validate_transformer_output(raw,'extract'),raw)
        consumed=raw[:,-16384:]
        self.assertEqual(consumed[0,0,0],5)
        self.assertIs(validate_transformer_output(consumed,'cached'),consumed)
        with self.assertRaises(ValueError):validate_transformer_output(raw,'cached')
        with self.assertRaises(ValueError):validate_transformer_output(raw[:,:100],'extract')

    def test_identity_distinguishes_dtype_shape_and_content(self):
        self.assertNotEqual(tensor_identity(torch.zeros(2)),tensor_identity(torch.zeros(1,2)))
        self.assertNotEqual(tensor_identity(torch.zeros(2)),tensor_identity(torch.ones(2)))


if __name__=='__main__':unittest.main()
