"""Explicit CPU reference for original C++ complete Whisper networks; no downloads."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import torch
from whisper.model import ModelDimensions, Whisper, MultiHeadAttention


def main(binary,exporter):
    spec=importlib.util.spec_from_file_location('whisper_export',exporter)
    conversion=importlib.util.module_from_spec(spec);spec.loader.exec_module(conversion)
    assert importlib.metadata.version('openai-whisper')=='20250625'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261010)
    MultiHeadAttention.use_sdpa=False
    records=[]
    with tempfile.TemporaryDirectory() as directory,torch.no_grad():
        root=Path(directory)
        for state,heads in [(8,2),(12,3),(16,4)]:
            dims=ModelDimensions(4,5,state,heads,2,17,7,state,heads,2)
            model=Whisper(dims).eval()
            for name,value in model.named_parameters():
                value.normal_(0,.15)
                if name.endswith('weight') and value.ndim==1:value.add_(1)
            original={name:value.clone() for name,value in model.state_dict().items()}
            for dtype in (torch.float32,torch.float16,torch.bfloat16):
                source=root/f'{state}-{dtype}.pt'
                torch.save(dict(dims=vars(dims),model_state_dict={name:value.to(dtype) for name,value in original.items()}),source)
                source_hash=hashlib.sha256(source.read_bytes()).hexdigest();destination=root/f'export-{state}-{dtype}'
                try:conversion.export(source,'0'*64,destination)
                except ValueError:pass
                else:raise AssertionError('accepted wrong source hash')
                assert not destination.exists()
                conversion.export(source,source_hash,destination)
                checkpoint=destination/'model.safetensors'
                try:conversion.export(source,source_hash,destination)
                except FileExistsError:pass
                else:raise AssertionError('replaced existing export')
                assert (destination/'dimensions.txt').read_text().split()==list(map(str,vars(dims).values()))
                model.load_state_dict({name:value.to(dtype).float() for name,value in original.items()})
                (root/'dims').write_text(' '.join(map(str,vars(dims).values())))
                for case in ('silence','seeded','impulse'):
                    mel=torch.zeros(1,4,10) if case!='seeded' else torch.randn(1,4,10)
                    if case=='impulse':mel[0,1,0]=1;mel[0,3,9]=-1
                    prompt=torch.tensor([[1,3,2]])
                    (root/'mel').write_bytes(mel.numpy().tobytes());(root/'prompt').write_bytes(struct.pack('<3I',*prompt[0].tolist()))
                    prefix=root/f'{state}-{dtype}-{case}'
                    result=json.loads(subprocess.check_output([str(binary),str(checkpoint.parent),checkpoint.name,str(root/'dims'),str(root/'mel'),str(root/'prompt'),str(prefix),'3','16']))
                    audio=model.encoder(mel);expected_logits=model.decoder(prompt,audio)[0,-1]
                    encoded=torch.frombuffer(bytearray(Path(str(prefix)+'.encoded.f32').read_bytes()),dtype=torch.float32).reshape(audio.shape)
                    logits=torch.frombuffer(bytearray(Path(str(prefix)+'.logits.f32').read_bytes()),dtype=torch.float32)
                    generated=[];tokens=prompt
                    for _ in range(3):
                        token=int(model.decoder(tokens,audio)[0,-1].argmax());generated.append(token)
                        if token==16:break
                        tokens=torch.cat((tokens,torch.tensor([[token]])),dim=1)
                    raw=Path(str(prefix)+'.tokens.u32').read_bytes();actual=list(struct.unpack(f'<{len(raw)//4}I',raw))
                    assert actual==generated,(actual,generated)
                    torch.testing.assert_close(encoded,audio,atol=2e-5,rtol=2e-5)
                    torch.testing.assert_close(logits,expected_logits,atol=2e-5,rtol=2e-5)
                    assert result['resident_bytes']==0 and not result['gpu_execution'] and not result['transcript']
                    records.append(dict(state=state,heads=heads,dtype=str(dtype),case=case,encoder_max_error=float((encoded-audio).abs().max()),decoder_max_error=float((logits-expected_logits).abs().max()),generated_tokens=actual,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    print(json.dumps(dict(reference='openai-whisper==20250625',torch_version=torch.__version__,cases=records,gpu_execution=False,full_size_checkpoint_validated=False),indent=2))

if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
