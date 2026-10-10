"""Explicit CPU full-codec reference; synthetic weights, no text generation."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

REPORT=sys.stdout
sys.stdout=sys.stderr

import torch
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder


def save(path,model):
    data=bytearray();header={}
    for name,value in model.state_dict().items():
        raw=value.detach().float().contiguous().numpy().tobytes();start=len(data);data.extend(raw)
        header['decoder.'+name]=dict(dtype='F32',shape=list(value.shape),data_offsets=[start,len(data)])
    encoded=json.dumps(header,separators=(',',':')).encode();path.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)


def main(binary):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261010)
    cfg=Qwen3TTSTokenizerV2DecoderConfig(latent_dim=8,codebook_dim=8,codebook_size=16,decoder_dim=32,hidden_size=8,intermediate_size=16,head_dim=4,num_attention_heads=2,num_key_value_heads=2,num_hidden_layers=2,num_quantizers=4,sliding_window=3)
    cfg._attn_implementation='eager'
    model=Qwen3TTSTokenizerV2Decoder(cfg).float().eval()
    with torch.no_grad():
        for name,value in model.named_parameters():
            value.normal_(0,.04)
            if 'cluster_usage' in name:value.fill_(1)
            elif 'norm.weight' in name:value.add_(1)
            elif name.endswith('.alpha') or name.endswith('.beta'):value.zero_()
    rows=[]
    with tempfile.TemporaryDirectory() as directory,torch.no_grad():
        root=Path(directory);save(root/'weights.safetensors',model)
        (root/'config').write_text('4 16 8 8 8 2 4 2 16 32 3')
        for frames in (1,2,4):
            codes=torch.randint(0,16,(1,4,frames))
            raw=codes[0].T.contiguous().numpy().astype('<u4').tobytes();(root/'codes').write_bytes(raw)
            output=root/f'{frames}.f32'
            result=json.loads(subprocess.check_output([str(binary),str(root),'weights.safetensors',str(root/'config'),str(root/'codes'),str(output)]))
            actual=torch.frombuffer(bytearray(output.read_bytes()),dtype=torch.float32)
            expected=model(codes).reshape(-1)
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
            assert actual.numel()==frames*1920 and result['resident_bytes']==0 and not result['full_tts_generation'] and not result['gpu_execution']
            rows.append(dict(frames=frames,samples=actual.numel(),maximum_absolute_error=float((actual-expected).abs().max()),rms_error=float(((actual-expected)**2).mean().sqrt()),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    print(json.dumps(dict(cases=rows,torch_version=torch.__version__,gpu_execution=False,real_checkpoint_validated=False),indent=2),file=REPORT)

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
