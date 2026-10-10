"""CPU reference: complete tiny talker and residual predictor versus cached C++."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
REPORT=sys.stdout
sys.stdout=sys.stderr
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration


def save(path,model,bfloat):
    header={};data=bytearray()
    for key,value in model.state_dict().items():
        value=value.detach().to(torch.bfloat16 if bfloat else torch.float32).contiguous()
        raw=value.view(torch.uint8).numpy().tobytes();start=len(data);data.extend(raw)
        header['talker.'+key]=dict(dtype='BF16' if bfloat else 'F32',shape=list(value.shape),data_offsets=[start,len(data)])
    encoded=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)


def reference(model,frames,variant):
    def text(ids):return model.text_projection(model.model.text_embedding(torch.tensor([ids])))
    def codec(ids):return model.model.codec_embedding(torch.tensor([ids]))
    pad=text([5]);bos=text([3]);end=text([4])
    speaker=19 if variant&2 else 20;language=17 if variant&2 else 18
    ids=[16,17,19,speaker,21] if variant&1 else [16,17,language,19,speaker,21]
    prefix=codec(ids)+torch.cat([pad]*(len(ids)-1)+[bos],dim=1)
    lead=[text([8,9,10])] if variant&2 else []
    inputs=torch.cat(lead+[text([0,1,2]),prefix,text([6,7])+codec([21]),end+codec([21]),pad+codec([22])],dim=1)
    result=[];previous=set()
    for frame in range(frames):
        hidden=model.model(inputs_embeds=inputs,use_cache=False).last_hidden_state[:,-1:]
        logits=model.codec_head(hidden)[0,0].clone()
        for token in previous:logits[token]=logits[token]*1.05 if logits[token]<0 else logits[token]/1.05
        first=int(logits[:16].argmax());previous.add(first);codes=[first];summed=codec([first]);sub=torch.cat([hidden,summed],dim=1)
        for group in range(3):
            projected=model.code_predictor.small_to_mtp_projection(sub)
            state=model.code_predictor.model(inputs_embeds=projected,use_cache=False).last_hidden_state[:,-1:]
            token=int(model.code_predictor.lm_head[group](state).argmax());codes.append(token)
            embedding=model.code_predictor.model.codec_embedding[group](torch.tensor([[token]]))
            summed=summed+embedding;sub=torch.cat([sub,embedding],dim=1)
        result.extend(codes);inputs=torch.cat([inputs,summed+pad],dim=1)
    return result


def main(binary):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);rows=[]
    cfg=Qwen3TTSTalkerConfig(vocab_size=24,text_vocab_size=32,text_hidden_size=8,hidden_size=8,intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=4,num_code_groups=4,rope_theta=1000000,rope_scaling={'rope_type':'default','mrope_section':[1,1,0],'interleaved':True},code_predictor_config=dict(vocab_size=16,hidden_size=4,intermediate_size=8,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=4,num_code_groups=4,rope_theta=1000000))
    cfg._attn_implementation='eager';cfg.code_predictor_config._attn_implementation='eager'
    for seed in (17,91,301):
        torch.manual_seed(seed);model=Qwen3TTSTalkerForConditionalGeneration(cfg).float().eval()
        with torch.no_grad():
            for name,value in model.named_parameters():
                value.normal_(0,.12)
                if 'norm.weight' in name:value.add_(1)
        original={k:v.clone() for k,v in model.state_dict().items()}
        for bfloat in (False,True):
            model.load_state_dict({k:v.to(torch.bfloat16).float() if bfloat else v for k,v in original.items()})
            with tempfile.TemporaryDirectory() as directory,torch.no_grad():
                root=Path(directory);save(root/'weights.safetensors',model,bfloat)
                for frames in (1,3,6):
                    for variant in range(4):
                        expected=reference(model,frames,variant);actual=json.loads(subprocess.check_output([binary,str(root),str(frames),str(variant)]))
                        assert actual['codes']==expected,(seed,bfloat,frames,variant,actual,expected)
                        assert actual['resident_bytes']==0 and not actual['stopped']
                        rows.append(dict(seed=seed,dtype='BF16' if bfloat else 'F32',frames=frames,variant=variant,codes=expected))
    print(json.dumps(dict(cases=rows,case_count=len(rows),gpu_execution=False,real_checkpoint_validated=False),indent=2),file=REPORT)
if __name__=='__main__':main(sys.argv[1])
