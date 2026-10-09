"""Pinned installed H3 equations, CPU only; consumes retained PR130 QKV evidence.
Usage: python video_rope_reference.py COMPONENT H3_SOURCE_DIRECTORY QKV_EVIDENCE NEW_DIRECTORY
No import of the SGLang package or GPU execution; only selected hash-checked AST nodes.
"""
import ast
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Tuple

import numpy as np
import torch

PINNED = {
    'attention.py':'a2d06ed14251937f98c8e903fb653282236222cc938569a37a1a3bb17b4579b0',
    'vit_utils.py':'fa142186b313ab33d049225d49de44e9f09527203cb1166a2b1f7f02667a9f25',
    'base_module.py':'14661aa1ef85c345eb1c64139d5640e98eab2dc35cd8b6e75151b51181d032d6',
    'vae_vit.py':'1e11a02564f2acbcdaed991a9fcb2e7060815b39c7cc4c56ed1bbcddbddb172d',
}
SELECTED = {'vit_utils.py':['_env_flag','create_token_ids','_rotate_half','_apply_rotary_pos_emb_impl'],
            'attention.py':['_vit_norm_input','_apply_qk_norm'], 'base_module.py':['RotaryEmbeddingND']}


def run(binary, source, qkv_root, output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    namespace=dict(torch=torch,nn=torch.nn,math=math,os=os,nullcontext=nullcontext,Tuple=Tuple)
    for filename,digest in PINNED.items():
        data=(source/filename).read_bytes();assert hashlib.sha256(data).hexdigest()==digest,filename
        names=SELECTED.get(filename,[])
        nodes=[node for node in ast.parse(data).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
        assert {node.name for node in nodes}==set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source/filename),'exec'),namespace)
    assert os.environ.get('MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM','1')=='1'
    prior=(qkv_root/'report.json').read_bytes()
    assert hashlib.sha256(prior).hexdigest()=='dd5484edd7f6a23ff44691e114ae55a7942574ca2490115127acc460961e023d'
    upstream=json.loads(prior);assert len(upstream['cases'])==24
    assert upstream['header_sha256']=='7bd6afbb1b2cfbf7c19901869eb224ec1b3e2a8cdbd8516d18f515c60f157c26'
    output.mkdir(parents=True,exist_ok=False)
    norm=torch.nn.RMSNorm(64,eps=1e-5,elementwise_affine=False)
    rotary=namespace['RotaryEmbeddingND'](48,100,n_dim=3,use_angle=True)
    cases=[]
    for case in upstream['cases']:
        tokens=case['tokens'];label=f"{tokens}-{case['pattern']}"
        with (qkv_root/label/'qkv.tensor').open('rb') as f:
            assert f.readline()==b'KADAN_H3_DECODER_QKV_V1\n'
            assert list(map(int,f.readline().split()))==[tokens,32,3,64]
            assert f.readline()==b'F32LE\n';payload=f.read()
        assert hashlib.sha256(payload).hexdigest()==case['sha256']
        qkv=torch.from_numpy(np.frombuffer(payload,dtype='<f4').copy().reshape(1,tokens,32,3,64))
        for axis in range(4 if tokens==8 else 3):
            dims=[2,2,2] if axis==3 else [1,1,1]
            if axis<3:dims[axis]=tokens
            ids=namespace['create_token_ids'](dims,torch.device('cpu'),torch.float32)
            # One zero-position suffix probe; this does not assemble learned suffix tokens.
            if tokens>1:ids[:,-1,:]=0
            table=rotary(ids)
            expected=qkv.clone()
            for kind in range(2):
                normalized=namespace['_apply_qk_norm'](norm,qkv[:,:,:,kind,:])
                expected[:,:,:,kind,:]=namespace['_apply_rotary_pos_emb_impl'](normalized,table)
            case_root=output/f'{label}-axis{axis}';case_root.mkdir()
            (case_root/'qkv.f32').write_bytes(payload)
            (case_root/'coordinates.f32').write_bytes(ids.numpy().astype('<f4').tobytes())
            completed=subprocess.run([str(binary),str(case_root/'qkv.f32'),str(case_root/'coordinates.f32'),str(case_root/'result.tensor')],check=True,capture_output=True,text=True)
            with (case_root/'result.tensor').open('rb') as f:
                assert f.readline()==b'KADAN_H3_QK_ROPE_V1\n'
                assert list(map(int,f.readline().split()))==[tokens,32,3,64]
                assert f.readline()==b'F32LE\n';actual_bytes=f.read()
            actual=np.frombuffer(actual_bytes,dtype='<f4').reshape(1,tokens,32,3,64)
            reference=expected.numpy();absolute=np.abs(actual-reference);bound=1e-5+1e-5*np.abs(reference)
            assert np.isfinite(actual).all() and np.all(absolute<=bound),(label,axis,float(absolute.max()))
            assert actual[:,:,:,2,:].tobytes()==qkv.numpy()[:,:,:,2,:].tobytes()
            cases.append(dict(tokens=tokens,pattern=case['pattern'],axis=axis,values=actual.size,
                max_absolute_error=float(absolute.max()),max_tolerance_ratio=float((absolute/bound).max()),
                input_sha256=case['sha256'],output_sha256=hashlib.sha256(actual_bytes).hexdigest(),stdout=completed.stdout.strip()))
    report=dict(reference='Hash-pinned installed H3 _apply_qk_norm + RotaryEmbeddingND + _apply_rotary_pos_emb_impl',
        source_sha256=PINNED,prior_qkv_report_sha256=hashlib.sha256(prior).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),reference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,cpu_threads=1,interop_threads=1,cpu_affinity=[cpu],gpu_execution=False,
        full_video_generation=False,production_fp16_parity=False,atol=1e-5,rtol=1e-5,cases=cases,
        compared_values=sum(c['values'] for c in cases),max_absolute_error=max(c['max_absolute_error'] for c in cases),
        max_tolerance_ratio=max(c['max_tolerance_ratio'] for c in cases),v_bitwise_unchanged=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
