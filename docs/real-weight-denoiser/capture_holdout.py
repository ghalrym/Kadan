"""Advance unchanged eager pipeline to one held-out step; write contexts, never weights."""
import argparse
from functools import wraps
import json
import os
from pathlib import Path
import time

import torch
from diffusers import QwenImage21Pipeline

from bf16_contracts import verify_capture, CAPTURE_MANIFEST
from capture import cpu, CaptureComplete, PROMPT, REVISION
from holdout_contracts import PROTOCOL, require_step
from holdout_packets import ContextWriter, estimate_context_bytes
from trace_binding import sha256


class StepCounter:
    def __init__(self,target):require_step(target);self.target=target;self.index=-1;self.timesteps=[]
    def observe(self,mode,timestep):
        self.index+=1
        if mode!=('extract' if self.index==0 else 'cached') or self.index>self.target:
            raise ValueError('Unexpected production forward/cache sequence')
        self.timesteps.append(timestep)
        return self.index==self.target


def check_state(actual,expected):
    if actual.keys()!=expected.keys():raise ValueError('Model weight schema differs from immutable reference')
    for name,value in actual.items():
        if not torch.equal(value.detach().cpu(),expected[name]):raise ValueError('Immutable weight differs: '+name)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=(20,39),required=True)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();root=Path('/capture');contexts=Path('/contexts')
    base=verify_capture(root);budget=estimate_context_bytes(root,base)
    prior_bytes=int(os.environ.get('KADAN_PRIOR_EVIDENCE_BYTES','0'))
    assert budget['projected_active_bytes']+prior_bytes<=budget['limit_bytes'], 'Full context shape budget does not fit'
    if args.plan:
        print(json.dumps(budget));return
    source=os.environ['KADAN_REVIEWED_COMMIT']
    writer=ContextWriter(contexts,sum(p.stat().st_size for p in root.iterdir())+prior_bytes)
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False
    pipeline=QwenImage21Pipeline.from_pretrained('/models/qwen-image-2.1-'+REVISION,
        local_files_only=True,torch_dtype=torch.bfloat16,use_safetensors=True)
    pipeline.vae.enable_tiling();pipeline.enable_model_cpu_offload(gpu_id=0)
    assert pipeline.transformer.config.causal_condition and len(pipeline.transformer.transformer_blocks)==32
    if dict(pipeline.scheduler.config)!=base['scheduler_config']:raise ValueError('Production scheduler configuration changed')
    counter=StepCounter(args.step);records=[];originals=[];tail={};started=time.monotonic()
    forward=pipeline.transformer.forward
    @wraps(forward)
    def transformer(*positional,**kwargs):
        counter.observe(kwargs.get('kv_cache_mode'),cpu(kwargs['timestep']).tolist())
        return forward(*positional,**kwargs)
    pipeline.transformer.forward=transformer
    def install(index,block):
        original=block.forward;originals.append((block,original))
        @wraps(original)
        def capture(*positional,**kwargs):
            if counter.index!=args.step:return original(*positional,**kwargs)
            assert not positional and len(records)==index and kwargs['kv_cache_mode']=='cached'
            base_row=base['blocks'][index]
            prior=torch.load(root/base_row['file'],weights_only=True,map_location='cpu',mmap=True)
            check_state(block.state_dict(),prior['state'])
            assert type(block.attn.processor).__qualname__==prior['processor_type']
            assert block.img_norm1.eps==prior['config']['eps']
            for name,field in [('hidden','hidden_states'),('modulation','modulation'),('rotary','rotary_emb')]:
                assert kwargs[field].shape==prior[name].shape
            for value,expected in zip(kwargs['layer_cache'].get(),prior['prefix']):assert value.shape==expected.shape
            del prior
            hidden=kwargs['hidden_states'];assert hidden.shape==(1,16384,4096) and hidden.dtype==torch.bfloat16
            assert bool(kwargs['target_token_mask'].all())
            mask=kwargs['attention_mask'];assert mask is None or (mask.dtype==torch.bool and mask.shape[1:3]==(1,1))
            packet=dict(hidden=cpu(hidden),modulation=cpu(kwargs['modulation']),rotary=cpu(kwargs['rotary_emb']),
                prefix=tuple(cpu(value) for value in kwargs['layer_cache'].get()),
                key_valid=cpu(mask[:,0,0] if mask is not None else None),step_index=args.step,base_packet_sha256=base_row['sha256'])
            result=original(*positional,**kwargs)
            packet['expected']=cpu(result)
            records.append(dict(index=index,**writer.write(f'block-{index:02}.context.pt',packet)))
            return result
        block.forward=capture
    for index,block in enumerate(pipeline.transformer.transformer_blocks):install(index,block)
    norm=pipeline.transformer.norm_out;project=pipeline.transformer.proj_out
    norm_forward,project_forward=norm.forward,project.forward
    def capture_norm(hidden,temb,mask):
        if counter.index==args.step:
            assert len(records)==32
            prior=torch.load(root/base['tail']['file'],weights_only=True,map_location='cpu',mmap=True)
            check_state(norm.state_dict(),prior['norm_state']);check_state(project.state_dict(),prior['projection_state'])
            assert float(pipeline.transformer.config.eps)==prior['eps']
            tail.update(temb=cpu(temb),target_mask=cpu(mask),step_index=args.step,base_packet_sha256=base['tail']['sha256'])
        return norm_forward(hidden,temb,mask)
    def capture_project(hidden):
        result=project_forward(hidden)
        if counter.index==args.step:
            tail['expected']=cpu(result)
            tail_row=writer.write('tail.context.pt',tail,tail=True)
            manifest=dict(protocol=PROTOCOL,step_index=args.step,source_commit=source,run_id=writer.run_id,
                base_manifest_sha256=CAPTURE_MANIFEST,budget=budget,image_digest=os.environ['KADAN_IMAGE_DIGEST'],blocks=records,tail=tail_row,seed=42,prompt=PROMPT,
                size=[2048,2048],steps=40,observed_timesteps=counter.timesteps,scheduler_config=dict(pipeline.scheduler.config),dtype='bfloat16',
                torch=torch.__version__,source_sha256=sha256(__file__),capture_elapsed_s=time.monotonic()-started,
                precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                    efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),math_sdp=torch.backends.cuda.math_sdp_enabled()))
            (contexts/'manifest.json').write_text(json.dumps(manifest,indent=2))
            raise CaptureComplete()
        return result
    norm.forward=capture_norm;project.forward=capture_project
    try:
        pipeline(prompt=PROMPT,width=2048,height=2048,num_inference_steps=40,num_images_per_prompt=1,
            generator=torch.Generator(device='cpu').manual_seed(42))
        raise AssertionError('Pipeline completed without selected-step capture')
    except CaptureComplete:
        assert len(records)==32 and counter.index==args.step and len(counter.timesteps)==args.step+1
    finally:
        for block,original in originals:block.forward=original
        pipeline.transformer.forward=forward;norm.forward=norm_forward;project.forward=project_forward
        pipeline.remove_all_hooks()


if __name__=='__main__':main()
