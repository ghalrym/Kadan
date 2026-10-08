"""Checkpoint fixtures and independent prefix equations; no real weights or GPU.
Reuse the reviewed stack equation oracle with BF16-decoded weights, serialize
canonical model files, and compare native CPU math plus fake-CUDA ownership.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile


def create(root, count=4):
    with contextlib.redirect_stdout(io.StringIO()):
        module = runpy.run_path(str(Path(__file__).with_name('stack_golden.py')))
    g = module['evaluate'].__globals__
    bf, f = g['bf'], g['f']
    # Independent reference: FP32 decode, BF16 weight, FP32/double reference sum,
    # BF16 activation. Non-power-of-two scales distinguish this from old fixtures.
    g['dot'] = lambda matrix, x: [bf(sum(bf(f(a))*b for a,b in zip(row,x))) for row in matrix]
    tensors = {}; declarations = {}
    def add(name, dtype, shape, data):
        assert len(data) == (2 if dtype == 'BF16' else 4 if dtype == 'F32' else 1) * prod(shape)
        tensors[name] = (dtype, shape, data)
    def prod(shape):
        n=1
        for v in shape:n*=v
        return n
    def flat(v):return sum(v, []) if isinstance(v[0], list) else v
    def dense(name, shape, values):
        add(name, 'BF16', shape, b''.join(struct.pack('<H',g['bits'](x)) for x in flat(values)))
    def projection(name, rows, cols, raw, scale, blocks=None, declaration=None):
        fp4=blocks is not None
        declarations[declaration or name]={'quant_algo':'W4A16_NVFP4' if fp4 else 'FP8', **({'group_size':16} if fp4 else {})}
        add(name+'.weight','U8' if fp4 else 'F8_E4M3',[rows,cols//2 if fp4 else cols],bytes(raw))
        add(name+'.weight_scale','F8_E4M3' if fp4 else 'F32',[rows,cols//16] if fp4 else [],bytes(blocks) if fp4 else struct.pack('<f',scale))
        if fp4:add(name+'.weight_scale_2','F32',[],struct.pack('<f',scale))
        add(name+'.input_scale','F32',[],struct.pack('<f',1.003))
    dense('model.language_model.embed_tokens.weight',[16,16],g['embedding'])
    dense('model.language_model.norm.weight',[16],g['final_norm'])
    projection('lm_head',16,16,[g['head_code'](r,j)|(g['head_code'](r,j+1)<<4) for r in range(16) for j in range(0,16,2)],.753,
               [0x20 if (0 if r==1 else r)%2==0 else 0x28 for r in range(16)])
    g['head']=[[bf(f(v*f(.753)/.75)) for v in row] for row in g['head']]
    for i in range(count):
        base=f'model.language_model.layers.{i}';full=i%4==3
        dense(base+'.input_layernorm.weight',[16],g['innorm']);dense(base+'.post_attention_layernorm.weight',[16],g['postnorm'])
        dense(base+'.mlp.gate.weight',[4,16],g['router']);dense(base+'.mlp.shared_expert_gate.weight',[1,16],g['shared_gate'])
        roles=[('self_attn.q_proj','fqg'),('self_attn.k_proj','fk'),('self_attn.v_proj','fv'),('self_attn.o_proj','fout')] if full else [('linear_attn.in_proj_qkv','lqkv'),('linear_attn.in_proj_z','lz'),('linear_attn.out_proj','lout')]
        for suffix,key in roles:
            matrix=g['base_matrices'][key];projection(base+'.'+suffix,len(matrix),len(matrix[0]),[g['encode'](v) for v in flat(matrix)],f(.503+.25*(i%4)))
        for suffix,key,shape in ([('self_attn.q_norm.weight','qn',[4]),('self_attn.k_norm.weight','kn',[4])] if full else [('linear_attn.conv1d.weight','conv',[12,1,3]),('linear_attn.in_proj_a.weight','la',[2,16]),('linear_attn.in_proj_b.weight','lb',[2,16]),('linear_attn.A_log','logs',[2]),('linear_attn.dt_bias','dt',[2]),('linear_attn.norm.weight','gate',[2])]):dense(base+'.'+suffix,shape,g[key])
        for e in range(5):
            middle=32 if e==4 else 16;family=base+('.mlp.shared_expert' if e==4 else '.mlp.experts');prefix=family if e==4 else family+f'.{e}'
            for role,suffix in enumerate(['gate_proj','up_proj','down_proj']):
                rows,cols=(16,middle) if role==2 else (middle,16)
                def code(r,j):return (r+2*j+3*e+role)%5+(8 if (r+j+e+role)%3==0 else 0)
                packed=[code(r,j)|(code(r,j+1)<<4) for r in range(rows) for j in range(0,cols,2)]
                blocks=[0x28 if (r+j+e+role)%2 else 0x20 for r in range(rows) for j in range(cols//16)]
                projection(prefix+'.'+suffix,rows,cols,packed,f(((3*e+role)%4+2)/4*f(.503+.125*(i%4))),blocks,None if e==4 else family)
    t=dict(model_type='qwen3_5_moe_text',dtype='bfloat16',hidden_act='silu',mamba_ssm_dtype='float32',attention_bias=False,attention_dropout=0,attn_output_gate=True,tie_word_embeddings=False,use_cache=True,hidden_size=16,vocab_size=16,num_hidden_layers=count,num_experts=4,num_experts_per_tok=2,moe_intermediate_size=16,shared_expert_intermediate_size=32,num_attention_heads=2,num_key_value_heads=1,head_dim=4,linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=2,linear_value_head_dim=2,linear_conv_kernel_dim=3,max_position_embeddings=8,bos_token_id=2,eos_token_id=15,rms_norm_eps=1e-6,partial_rotary_factor=.5,full_attention_interval=4,layer_types=['full_attention' if i%4==3 else 'linear_attention' for i in range(count)],rope_parameters=dict(rope_type='default',mrope_interleaved=True,partial_rotary_factor=.5,rope_theta=10000,mrope_section=[1,0,0]))
    config=dict(model_type='qwen3_5_moe',architectures=['Qwen3_5MoeForConditionalGeneration'],tie_word_embeddings=False,text_config=t,quantization_config=dict(quant_method='modelopt',quant_algo='MIXED_PRECISION',producer={'name':'modelopt'},quantized_layers=declarations))
    (root/'config.json').write_text(json.dumps(config));(root/'generation_config.json').write_text(json.dumps({'eos_token_id':[15,14]}))
    header={};payload=bytearray()
    for name,(dtype,shape,data) in tensors.items():
        header[name]=dict(dtype=dtype,shape=shape,data_offsets=[len(payload),len(payload)+len(data)]);payload.extend(data)
    raw=json.dumps(header).encode();(root/'fixture.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+payload)
    (root/'model.safetensors.index.json').write_text(json.dumps(dict(metadata={'total_size':len(payload)},weight_map={n:'fixture.safetensors' for n in tensors})))
    # Prefix recomputation uses independent equations and immutable checkpoint values.
    expected=bytearray();capture=bytearray(struct.pack("<4I",0x4b4d4331,1,16,3))
    for end in range(1,4):
        g['inputs']=[g['embedding'][t] for t in [2,7,11][:end]];records=[]
        for i in range(count):
            scale=f(.503+.25*(i%4));g['moe_scale']=f(.503+.125*(i%4))
            for name,original in g['base_matrices'].items():g[name]=[[bf(f(v*scale)) for v in row] for row in original]
            attn=g['attention_full']() if i%4==3 else g['attention_linear']()
            mixed=[g['mixture'](g['norm'](row[0],g['postnorm'])) for row in attn]
            g['inputs']=[[bf(a+b) for a,b in zip(row[0],m)] for row,m in zip(attn,mixed)]
            records.append((g['inputs'][-1],attn[-1][1],attn[-1][2]))
        logits=g['dot'](g['head'],g['norm'](g['inputs'][-1],g['final_norm']));selected=max(range(16),key=lambda j:(logits[j],-j))
        expected.extend(struct.pack('<I16f',selected,*logits))
        capture.extend(struct.pack('<4I16f',[2,7,11][end-1],selected,int(selected in (14,15)),end,*logits))
        for i,(out,first,second) in enumerate(records):
            expected.extend(struct.pack('<16f',*out));expected.extend(struct.pack('<'+'H'*len(first),*first));expected.extend(struct.pack('<'+('H' if i%4==3 else 'f')*len(second),*second))
    (root/'expected.bin').write_bytes(expected)
    capture.extend(struct.pack('<I',0x444f4e45));(root/'reference.capture').write_bytes(capture)
    return header,len(raw)

if __name__=='__main__':
    unit,runtime,harness,compare=sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix='kadan-model-fixture-') as directory:
        root=Path(directory)
        for layers in (4,8,40):
            create(root,layers)
            for binary in (unit,runtime):subprocess.run([binary,str(root)],check=True,timeout=25)

        # Malformed generation controls and payload errors after partial upload.
        header,header_bytes=create(root)
        original_generation=(root/'generation_config.json').read_bytes()
        for eos in ([],[14,14],[16],[-1],"14",list(range(17))):
            (root/'generation_config.json').write_text(json.dumps({'eos_token_id':eos}))
            for binary in (unit,runtime):subprocess.run([binary,str(root),'reject'],check=True,timeout=25)
        (root/'generation_config.json').write_bytes(original_generation)
        path=root/'fixture.safetensors';original=path.read_bytes()
        corruptions=[('model.language_model.embed_tokens.weight',b'\x80\x7f'),
                     ('model.language_model.layers.0.linear_attn.in_proj_qkv.weight',b'\x7f'),
                     ('model.language_model.layers.3.mlp.shared_expert.down_proj.weight_scale_2',struct.pack('<f',float('nan'))),
                     ('lm_head.weight_scale',b'\xff'),('lm_head.input_scale',struct.pack('<f',0)),
                     ('lm_head.weight_scale_2',struct.pack('<f',-1))]
        for name,data in corruptions:
            changed=bytearray(original);at=8+header_bytes+header[name]['data_offsets'][0];changed[at:at+len(data)]=data;path.write_bytes(changed)
            for binary in (unit,runtime):subprocess.run([binary,str(root),'reject'],check=True,timeout=25)
            if 'shared_expert.down_proj' in name:subprocess.run([runtime,str(root),'reject-cleanup'],check=True,timeout=25)
        path.write_bytes(original)

        # Exercise the actual standalone harness against the CPU fake runtime.
        output=root/'capture.bin'
        command=[harness,'--execute',str(root),'0','8',str(300*1024**2),str(2*1024**2),'512','5',str(output),'2','7']
        subprocess.run(command,check=True,timeout=10)
        data=output.read_bytes();assert struct.unpack_from('<4I',data)==(0x4b4d4331,1,16,2)
        assert len(data)==16+2*(16+16*4)+4 and struct.unpack_from('<I',data,len(data)-4)[0]==0x444f4e45
        baseline=root/'baseline.bin';baseline.write_bytes(data)
        subprocess.run([compare,str(baseline),str(output),'0','0'],check=True,timeout=5)
        changed=bytearray(data);struct.pack_into('<f',changed,32+4,-.5);output.write_bytes(changed)
        assert subprocess.run([compare,str(baseline),str(output),'0','0'],capture_output=True,timeout=5).returncode==1
        output.write_bytes(data[:-4])
        assert subprocess.run([compare,str(baseline),str(output),'0','0'],capture_output=True,timeout=5).returncode==1
        changed=bytearray(data);struct.pack_into('<I',changed,8,2**32-1);output.write_bytes(changed)
        assert subprocess.run([compare,str(baseline),str(output),'0','0'],capture_output=True,timeout=5).returncode==1
        output.write_bytes(data)
        assert subprocess.run(command,capture_output=True,timeout=10).returncode==1  # O_EXCL
        output.unlink()
        invalid=command.copy();invalid[-1]='16'
        assert subprocess.run(invalid,capture_output=True,timeout=10).returncode==1 and not output.exists()
        stalled=command.copy();stalled[8]='1'
        result=subprocess.run(stalled,env={**os.environ,'KADAN_TEST_STALLED_RUNTIME':'1'},capture_output=True,timeout=5)
        assert result.returncode==124 and output.read_bytes()==b''
