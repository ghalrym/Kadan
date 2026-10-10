import sys
REPORT=sys.stdout;sys.stdout=sys.stderr
import hashlib,json,struct,subprocess,tempfile
from pathlib import Path
import torch
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder

torch.set_num_threads(1);torch.set_num_interop_threads(1)
binary=Path(sys.argv[1]).resolve();checkpoint=Path(sys.argv[2]).resolve();before=checkpoint.stat()
config=json.loads((checkpoint.parent/'config.json').read_text())['decoder_config'];cfg=Qwen3TTSTokenizerV2DecoderConfig(**config);cfg._attn_implementation='eager'
model=Qwen3TTSTokenizerV2Decoder(cfg).float().eval();parameters=dict(model.named_parameters());digests={}
with checkpoint.open('rb') as source,torch.no_grad():
 length=struct.unpack('<Q',source.read(8))[0];assert length<1024*1024;encoded=source.read(length);header=json.loads(encoded)
 for name,value in parameters.items():
  item=header['decoder.'+name];assert item['dtype']=='F32' and list(value.shape)==item['shape'];start,end=item['data_offsets'];source.seek(8+length+start);raw=bytearray(source.read(end-start));assert len(raw)==end-start
  value.copy_(torch.frombuffer(raw,dtype=torch.float32).reshape(value.shape));digests[name]=hashlib.sha256(raw).hexdigest()
 codes=(torch.arange(16).reshape(1,16,1)*97+31)%2048
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory);(root/'config').write_text('16 2048 512 1024 512 16 64 8 1024 1536 72');(root/'codes').write_bytes(codes[0].T.contiguous().numpy().astype('<u4').tobytes())
  result=json.loads(subprocess.check_output([str(binary),str(checkpoint.parent),checkpoint.name,str(root/'config'),str(root/'codes'),str(root/'output')]))
  actual=torch.frombuffer(bytearray((root/'output').read_bytes()),dtype=torch.float32);expected=model(codes).reshape(-1)
  error=(actual-expected).abs();report={'samples':actual.numel(),'maximum_absolute_error':float(error.max()),'rms_error':float((error**2).mean().sqrt()),'waveform_peak':float(expected.abs().max()),'header_sha256':hashlib.sha256(encoded).hexdigest(),'selected_tensor_count':len(parameters),'selected_tensor_sha256':digests,'native_result':result,'gpu_execution':False,'text_to_speech_validated':False,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
  print(json.dumps(report,indent=2),file=REPORT,flush=True)
  torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
 assert (before.st_ino,before.st_size,before.st_mtime_ns)==(checkpoint.stat().st_ino,checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
