"""Capture real first-cached-step blocks from the unchanged eager pipeline.

This is an isolated benchmark process, never an API hook. Deliberately stop after
block31's cached output; no VAE/full40-step result is claimed.
"""
import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path
import time

import torch
from diffusers import QwenImage21Pipeline

REVISION = 'd26bb61231c349cf6b7896fa83353113880e1ba3'
PROMPT = 'A single red apple on a plain white table, soft natural daylight, realistic still-life photograph.'
LIMIT = 32 * 1024**3


class CaptureComplete(Exception):
    pass


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def cpu(value):
    return value.detach().to('cpu').clone().contiguous() if value is not None else None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    model = args.model.resolve()
    assert model.is_relative_to('/models') and REVISION in model.name, 'Use exact pinned checkpoint directory'
    root = Path('/capture')
    assert not list(root.iterdir()), 'Capture destination must be empty'
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline = QwenImage21Pipeline.from_pretrained(str(model), local_files_only=True,
        torch_dtype=torch.bfloat16, use_safetensors=True)
    assert pipeline.transformer.config.causal_condition and len(pipeline.transformer.transformer_blocks) == 32
    pipeline.vae.enable_tiling()
    pipeline.enable_model_cpu_offload(gpu_id=0)
    records = []
    calls = {'extract': 0, 'cached': 0}
    captured_timestep = []
    transformer_forward = pipeline.transformer.forward
    @wraps(transformer_forward)
    def transformer_call(*args, **kwargs):
        mode = kwargs.get('kv_cache_mode')
        assert mode in calls, 'Expected pinned eager cache protocol'
        calls[mode] += 1
        if mode == 'cached':
            assert calls == {'extract': 1, 'cached': 1}, 'Unexpected guidance/cache call sequence'
            captured_timestep.extend(kwargs['timestep'].detach().cpu().tolist())
        return transformer_forward(*args, **kwargs)
    pipeline.transformer.forward = transformer_call
    originals = []
    started = time.monotonic()
    def install(index, block):
        original = block.forward
        originals.append((block, original))
        @wraps(original)
        def captured(*positional, **kwargs):
            if kwargs.get('kv_cache_mode') != 'cached':
                return original(*positional, **kwargs)
            assert not positional and len(records) == index, 'Unexpected cached call order/signature'
            assert time.monotonic()-started < 900
            hidden = kwargs['hidden_states']
            assert hidden.shape == (1,16384,4096) and hidden.dtype == torch.bfloat16
            assert bool(kwargs['target_token_mask'].all()), 'Only cached target rows may be replayed'
            mask = kwargs['attention_mask']
            if mask is not None:
                assert mask.dtype == torch.bool and mask.ndim == 4 and mask.shape[1:3] == (1,1)
            prefix = kwargs['layer_cache'].get()
            packet = dict(processor_type=type(block.attn.processor).__qualname__,hidden=cpu(hidden), modulation=cpu(kwargs['modulation']),
                rotary=cpu(kwargs['rotary_emb']), key_valid=cpu(mask[:,0,0] if mask is not None else None),
                prefix=tuple(cpu(value) for value in prefix),
                state={name:cpu(value) for name,value in block.state_dict().items()},
                config=dict(dim=4096,num_attention_heads=32,attention_head_dim=128,
                    mlp_ratio=3,eps=block.img_norm1.eps))
            output = original(*positional, **kwargs)
            packet['expected'] = cpu(output)
            estimate=sum(v.numel()*v.element_size() for v in packet['state'].values())
            estimate+=sum(v.numel()*v.element_size() for v in (packet['hidden'],packet['expected'],packet['modulation'],packet['rotary'],*packet['prefix']))
            used=sum(p.stat().st_size for p in root.iterdir())
            assert used + estimate + 16*1024**2 < LIMIT, 'Capture disk envelope exhausted'
            partial=root/f'block-{index:02}.partial'
            torch.save(packet,partial)
            path=partial.with_suffix('.pt'); partial.rename(path)
            assert sum(p.stat().st_size for p in root.iterdir()) < LIMIT
            records.append(dict(index=index,file=path.name,sha256=digest(path),bytes=path.stat().st_size))
            print(json.dumps(dict(captured=index,bytes=path.stat().st_size)),flush=True)
            return output
        block.forward = captured
    for index,block in enumerate(pipeline.transformer.transformer_blocks):
        install(index,block)
    tail = {}
    norm = pipeline.transformer.norm_out
    project = pipeline.transformer.proj_out
    norm_forward, project_forward = norm.forward, project.forward
    def capture_norm(hidden, temb, mask):
        if len(records) == 32:
            tail.update(temb=cpu(temb), target_mask=cpu(mask),
                norm_state={name:cpu(value) for name,value in norm.state_dict().items()},
                projection_state={name:cpu(value) for name,value in project.state_dict().items()},
                eps=float(pipeline.transformer.config.eps))
        return norm_forward(hidden, temb, mask)
    def capture_project(hidden):
        output = project_forward(hidden)
        if tail:
            tail['expected'] = cpu(output)
            path=root/'tail.pt'
            torch.save(tail,path)
            assert sum(p.stat().st_size for p in root.iterdir()) < LIMIT
            raise CaptureComplete()
        return output
    norm.forward, project.forward = capture_norm, capture_project
    try:
        pipeline(prompt=PROMPT,width=2048,height=2048,num_inference_steps=40,
            num_images_per_prompt=1,generator=torch.Generator(device='cpu').manual_seed(42))
        raise AssertionError('Pipeline completed without expected cached capture')
    except CaptureComplete:
        assert len(records)==32
        manifest=dict(format=1,origin='real-eager',revision=REVISION,prompt=PROMPT,seed=42,
            size=[2048,2048],steps=40,cached_step=1,timestep=captured_timestep,scheduler_config=dict(pipeline.scheduler.config),dtype='bfloat16',blocks=records,tail=dict(file='tail.pt',sha256=digest(root/'tail.pt')),
            capture_elapsed_s=time.monotonic()-started,torch=torch.__version__,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    finally:
        for block,original in originals: block.forward=original
        norm.forward, project.forward = norm_forward, project_forward
        pipeline.transformer.forward = transformer_forward
        pipeline.remove_all_hooks()


if __name__ == '__main__':
    main()
