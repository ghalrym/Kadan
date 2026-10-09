"""One independent full pipeline per process; no reference values enter model state."""
from datetime import timedelta
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from diffusers import QwenImage21Pipeline

from bf16_contracts import ULYSSES_SOURCE
from capture import PROMPT, REVISION
from trajectory_adapter import UlyssesAdapter
from trajectory_contracts import PROTOCOL, SETTINGS, CRITERIA, StepOrder, verify_run
from trajectory_io import ArtifactWriter, tensor_identity, measure, normalized_rgb
from trace_binding import sha256

OUT=Path('/evidence');REFERENCE=Path('/reference')


def main():
    wall_started=time.monotonic();case=os.environ['KADAN_TRAJECTORY_CASE'];candidate=case=='candidate'
    rank=int(os.environ.get('LOCAL_RANK','0'));commit=os.environ['KADAN_REVIEWED_COMMIT']
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    assert quota!='max' and int(quota)<=2*int(period)
    torch.cuda.set_device(rank);torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(rank).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False
    control=None
    if candidate:
        dist.init_process_group('nccl',timeout=timedelta(seconds=120),device_id=torch.device('cuda',rank))
        control=dist.new_group(backend='gloo',timeout=timedelta(seconds=120))
    def phase(name):
        p=OUT/f'phase-rank-{rank}.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(phase=name,rank=rank,unix_time=time.time())));tmp.replace(p)
    def consensus(ok,label):
        if candidate:
            values=[None,None];dist.all_gather_object(values,dict(ok=bool(ok),label=label),group=control)
            if any(v['label']!=label or not v['ok'] for v in values):raise AssertionError('Rank gate failed: '+label)
        elif not ok:raise AssertionError('Gate failed: '+label)
    def identical_across_ranks(value,label):
        if candidate:
            values=[None,None];dist.all_gather_object(values,value,group=control)
            consensus(values[0]==values[1],label)
    writer=ArtifactWriter(OUT);reports=[];order=StepOrder();adapter=None;pipeline=None
    status=dict(protocol=PROTOCOL,case=case,status='incomplete',source_commit=commit,settings=SETTINGS,criteria=CRITERIA,production_activation=False)
    def save_status(): (OUT/f'verdict-rank-{rank}.json').write_text(json.dumps(status,indent=2))
    save_status()
    try:
        phase('verify-reference')
        if case!='reference':verify_run(REFERENCE,'reference',commit)
        assert sha256('/app/api/inference/image/parallel.py')==ULYSSES_SOURCE
        phase('hash-checkpoint')
        model_root=Path('/models/qwen-image-2.1-'+REVISION)
        checkpoint_files=None
        if rank==0:
            checkpoint_files=[dict(file=str(p.relative_to(model_root)),sha256=sha256(p),bytes=p.stat().st_size)
                for p in sorted(model_root.rglob('*')) if p.is_file() and not any(part.startswith('.') for part in p.relative_to(model_root).parts)]
            assert checkpoint_files and any(row['file'].endswith('.safetensors') for row in checkpoint_files)
        if candidate:
            payload=[checkpoint_files];dist.broadcast_object_list(payload,src=0,group=control);checkpoint_files=payload[0]
        phase('load-pipeline')
        pipeline=QwenImage21Pipeline.from_pretrained('/models/qwen-image-2.1-'+REVISION,local_files_only=True,torch_dtype=torch.bfloat16,use_safetensors=True)
        assert pipeline.transformer.config.causal_condition and len(pipeline.transformer.transformer_blocks)==32
        pipeline.vae.enable_tiling();pipeline.enable_model_cpu_offload(gpu_id=rank)
        if candidate:adapter=UlyssesAdapter(pipeline.transformer,rank,control);adapter.install()
        source_identity=dict(pipeline=sha256(inspect.getfile(QwenImage21Pipeline)),transformer=sha256(inspect.getfile(type(pipeline.transformer))),
            scheduler=sha256(inspect.getfile(type(pipeline.scheduler))),vae=sha256(inspect.getfile(type(pipeline.vae))))
        (OUT/f'cpu-threads-rank-{rank}.json').write_text(json.dumps(dict(intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads(),cpu_max=f'{quota} {period}',affinity_cpus=len(os.sched_getaffinity(0)))))
        def preserve_failure(label,actual,reference=None):
            path=OUT/f'failed-rank-{rank}.pt'
            if not path.exists():
                values=dict(label=label,actual=actual.detach().cpu().clone())
                if reference is not None:values['reference']=reference.detach().cpu().clone()
                size=sum(v.numel()*v.element_size() for v in values.values() if isinstance(v,torch.Tensor))
                if size>128*1024**2:raise RuntimeError('Failure artifact exceeds fixed per-rank reserve')
                torch.save(values,path)
        inputs={};timesteps=[];mode={'value':None};originals=[];request_started=None
        def record_tensor(name,value,kind='latent',exact=False):
            phase('audit:'+name);actual=value.detach().cpu().contiguous()
            if rank==0:writer.tensor(name,actual)
            if case!='reference':
                reference=torch.load(REFERENCE/name,weights_only=True,map_location='cpu',mmap=True)
                row=measure(actual,reference,kind)
                if exact:row['passed']=actual.dtype==reference.dtype and torch.equal(actual,reference)
                row.update(name=name,rank=rank);reports.append(row)
                with (OUT/f'comparisons-rank-{rank}.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                if not row['passed']:preserve_failure(name,actual,reference)
                consensus(row['passed'],name)
            else:consensus(bool(torch.isfinite(actual).all()),name)
            return actual
        prepare=pipeline.prepare_latents;originals.append((pipeline,'prepare_latents',prepare))
        @wraps(prepare)
        def prepare_latents(*args,**kwargs):
            phase('prepare-latents');result=prepare(*args,**kwargs)
            assert result[1] is None and result[0].shape==(1,16384,64) and result[0].dtype==torch.bfloat16
            record_tensor('initial.pt',result[0],exact=True)
            identical_across_ranks(tensor_identity(result[0]),'candidate-initial')
            return result
        pipeline.prepare_latents=prepare_latents
        forward=pipeline.transformer.forward;originals.append((pipeline.transformer,'forward',forward))
        @wraps(forward)
        def transformer(*args,**kwargs):
            index=order.next;current=kwargs['kv_cache_mode'];order.prediction(index,current);mode['value']=current
            if adapter:adapter.mode=current;adapter.step=index
            if index==0:
                inputs.update(settings=SETTINGS,sources=source_identity,checkpoint_files=checkpoint_files,scheduler_config=dict(pipeline.scheduler.config),
                    timesteps=pipeline.scheduler.timesteps.detach().cpu().tolist(),
                    conditioning={key:tensor_identity(kwargs[key]) for key in ('encoder_hidden_states','encoder_hidden_states_mask','img_mask')},
                    image_shapes=kwargs['img_shapes'],precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,cudnn_tf32=torch.backends.cudnn.allow_tf32),
                    float_image=dict(space='pipeline display RGB interpreted as sRGB; no additional transfer/profile conversion',layout='NHWC',dtype='float32',range=[0,1],point='postprocess numpy output before uint8 rounding',raw_vae_nonfinite='reject before normalization or clipping'))
                # Canonicalize tuples/config values through JSON before identity checks.
                canonical=json.loads(json.dumps(inputs))
                identical_across_ranks(canonical,'candidate-conditioning')
                if rank==0:writer.json('input-identity.json',canonical)
                if case!='reference':consensus(canonical==json.loads((REFERENCE/'input-identity.json').read_text()),'input-identity')
            phase(f'transformer:{current}:{index}')
            result=forward(*args,**kwargs)
            assert isinstance(result,tuple) and len(result)==1
            output=result[0] if adapter is None else adapter.gather(result[0])
            assert output.shape==(1,16384,64)
            return (output,)
        pipeline.transformer.forward=transformer
        scheduler_step=pipeline.scheduler.step;originals.append((pipeline.scheduler,'step',scheduler_step))
        @wraps(scheduler_step)
        def step(prediction,timestep,latent,*args,**kwargs):
            index=order.scheduler();timesteps.append(float(timestep))
            assert prediction.shape==latent.shape==(1,16384,64)
            record_tensor(f'prediction-{index:02}.pt',prediction)
            identical_across_ranks(tensor_identity(prediction),f'candidate-prediction-{index}')
            phase(f'scheduler:{index}');result=scheduler_step(prediction,timestep,latent,*args,**kwargs)
            record_tensor(f'latent-{index:02}.pt',result[0])
            identical_across_ranks(tensor_identity(result[0]),f'candidate-latent-{index}')
            # Return only this path's own scheduler output, never the loaded reference.
            return result
        pipeline.scheduler.step=step
        postprocess=pipeline.image_processor.postprocess;originals.append((pipeline.image_processor,'postprocess',postprocess))
        @wraps(postprocess)
        def image_postprocess(image,*args,**kwargs):
            phase('raw-vae-finite-check');order.complete()
            finite=bool(torch.isfinite(image).all())
            if not finite:preserve_failure('raw-vae-before-clipping',image)
            consensus(finite,'raw-vae-before-clipping')
            assert not args and kwargs.get('output_type')=='pil'
            phase('image-postprocess');rgb=normalized_rgb(image,postprocess)
            assert rgb.shape==(1,2048,2048,3) and rgb.dtype==np.float32
            record_tensor('float-rgb.pt',torch.from_numpy(rgb),'float')
            images=pipeline.image_processor.numpy_to_pil(rgb)
            pixels=torch.from_numpy(np.asarray(images[0]).copy())
            assert pixels.dtype==torch.uint8 and pixels.shape==(2048,2048,3)
            record_tensor('pixels.pt',pixels,'pixel')
            if rank==0:
                writer.admit(16*1024**2);images[0].save(OUT/'output.png');assert (OUT/'output.png').stat().st_size<=16*1024**2;writer.record('output.png')
            return images
        pipeline.image_processor.postprocess=image_postprocess
        torch.cuda.synchronize();request_started=time.monotonic();phase('request-start-text-encoding')
        pipeline(prompt=PROMPT,width=2048,height=2048,num_inference_steps=40,num_images_per_prompt=1,
            generator=torch.Generator(device='cpu').manual_seed(42),true_cfg_scale=1.0,use_kv_cache=True,output_type='pil')
        torch.cuda.synchronize();request_elapsed=time.monotonic()-request_started;order.complete()
        assert len(timesteps)==40 and timesteps==inputs['timesteps']
        timing=dict(request_instrumented_seconds=request_elapsed,main_to_completion_seconds=time.monotonic()-wall_started,
            includes='text encoding, offload, independent prefill, all denoising/scheduler steps, VAE, image preparation, disk writes and audits',
            excludes='container startup and API restoration; reported in launcher pause evidence',speedup_claim=False)
        if candidate:
            times=[None,None];dist.all_gather_object(times,timing,group=control)
            timing['slower_rank_request_seconds']=max(t['request_instrumented_seconds'] for t in times)
        (OUT/f'timing-rank-{rank}.json').write_text(json.dumps(timing,indent=2))
        expected_reports=0 if case=='reference' else 83
        assert len(reports)==expected_reports and all(row['passed'] for row in reports)
        report_rows=[]
        if case!='reference' and rank==0:
            for report_rank in range(2 if candidate else 1):
                path=OUT/f'comparisons-rank-{report_rank}.jsonl'
                report_rows.append(dict(file=path.name,sha256=sha256(path)))
        status.update(comparisons_count=len(reports),comparison_reports=report_rows,status='passed',completed_steps=40,raw_vae_finite=True,timesteps=timesteps,artifacts=writer.rows if rank==0 else [],reference_manifest_sha256=None if case=='reference' else sha256(REFERENCE/'manifest.json'))
        save_status()
        if rank==0:(OUT/'manifest.json').write_text(json.dumps(status,indent=2))
    except BaseException as exc:
        status.update(status='failed' if isinstance(exc,AssertionError) else 'incomplete',error=dict(type=type(exc).__name__,message=str(exc)[:2000]))
        save_status()
        raise
    finally:
        if adapter:adapter.close()
        if pipeline:pipeline.remove_all_hooks()
        if candidate and dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    with torch.no_grad():main()
