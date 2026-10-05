"""Compare tiny CPU Qwen outputs between two installed dependency environments.

This is an explicit developer check, not part of CI's single-environment smoke.
Run both commands from the repository root with the same torch CPU version:

    OLD_PYTHON api/qwen_tts_compat/compare_versions.py --record /tmp/qwen-reference
    NEW_PYTHON api/qwen_tts_compat/compare_versions.py /tmp/qwen-reference

OLD_PYTHON should contain official Qwen commit
022e286b98fbec7e1e916cb940cdf532cd9f488e with Transformers 4.57.3;
NEW_PYTHON should contain Kadan's patched package and pinned Transformers.
The reference directory contains locally generated tiny random tensors only.
Use a fresh directory when recording a baseline. No network or model downloads.
"""
import argparse
from importlib.metadata import version
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--record', action='store_true')
parser.add_argument('reference', type=Path)
args = parser.parse_args()
old, fixture = args.record, args.reference
if old:
    fixture.mkdir(parents=True, exist_ok=False)
print({name: version(name) for name in ('qwen-tts', 'transformers', 'torch')})
torch.manual_seed(42)
def weights(name, model):
    path=fixture / (name+'.pt')
    if old: torch.save(model.state_dict(),path)
    else: model.load_state_dict(torch.load(path,weights_only=True),strict=True)
    return model
results={}

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel
config=Qwen3TTSTalkerConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,text_hidden_size=16,text_vocab_size=32,pad_token_id=None,rope_scaling={'rope_type':'default','mrope_section':[1,1,2],'interleaved':False},code_predictor_config={'vocab_size':32,'hidden_size':16,'intermediate_size':32,'num_hidden_layers':1,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':8})
model=weights('model',Qwen3TTSTalkerModel(config).eval())
print('Tiny talker constructed')
with torch.inference_mode():
 out=model(inputs_embeds=model.codec_embedding(torch.tensor([[1,2]])),use_cache=True)
 results['forward']=out.last_hidden_state
 print('Tiny talker forward',out.last_hidden_state.shape)
 out2=model(inputs_embeds=model.codec_embedding(torch.tensor([[3]])),past_key_values=out.past_key_values,use_cache=True)
 results['cached']=out2.last_hidden_state
 print('Tiny talker cached forward',out2.last_hidden_state.shape,'cache',out2.past_key_values.get_seq_length())
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,Qwen3TTSTalkerForConditionalGeneration
config.num_code_groups=3
config.code_predictor_config.num_code_groups=3
config.code_predictor_config.pad_token_id=None
predictor=weights('predictor',Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(config.code_predictor_config,config).eval())
with torch.inference_mode():
 ids=predictor.generate(inputs_embeds=torch.randn(1,2,16),max_new_tokens=2,do_sample=False)
 results['predictor']=ids
 print('Tiny predictor generate',ids.shape)
talker=weights('talker',Qwen3TTSTalkerForConditionalGeneration(config).eval())
with torch.inference_mode():
 generated=talker.generate(inputs_embeds=torch.randn(1,3,16),trailing_text_hidden=torch.randn(1,2,16),tts_pad_embed=torch.randn(1,1,16),max_new_tokens=3,do_sample=False,subtalker_dosample=False,return_dict_in_generate=True,output_hidden_states=True)
 results['generation']=generated.sequences
 print('Tiny talker generate',generated.sequences.shape)
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
encoder=dict(hidden_size=16,num_filters=8,num_residual_layers=1,upsampling_ratios=[2,2],num_hidden_layers=1,intermediate_size=32,num_attention_heads=2,num_key_value_heads=2,head_dim=8,codebook_size=16,codebook_dim=16,num_quantizers=2,num_semantic_quantizers=1,vector_quantization_hidden_dimension=16,upsample_groups=1,use_cache=False)
decoder=dict(codebook_size=16,codebook_dim=16,hidden_size=16,latent_dim=16,num_attention_heads=2,num_key_value_heads=2,intermediate_size=32,num_hidden_layers=1,num_quantizers=2,upsample_rates=[2],upsampling_ratios=[2,2],decoder_dim=16,sliding_window=8)
tok=Qwen3TTSTokenizerV2Model(Qwen3TTSTokenizerV2Config(encoder_config=encoder,decoder_config=decoder,encoder_valid_num_quantizers=2,decode_upsample_rate=8,encode_downsample_rate=8)).eval()
weights('tokenizer',tok)
print('Tiny tokenizer constructed')
with torch.inference_mode():
 encoded=tok.encode(torch.randn(1,64),torch.ones(1,64,dtype=torch.bool))
 results['encode']=encoded.audio_codes[0]
 print('Tiny tokenizer encode',encoded.audio_codes[0].shape)
 decoded=tok.decode(encoded.audio_codes[0].unsqueeze(0))
 results['decode']=decoded.audio_values[0]
 print('Tiny tokenizer decode',decoded.audio_values[0].shape)

if old: torch.save(results,fixture/'results.pt')
else:
 reference=torch.load(fixture/'results.pt',weights_only=True)
 for name,result in results.items():
  torch.testing.assert_close(result,reference[name],rtol=1e-4,atol=1e-5)
  print('Cross-version output matched:',name)
from transformers import Wav2Vec2FeatureExtractor
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
if old:
 tok.save_pretrained(fixture/'tokenizer_saved')
 Wav2Vec2FeatureExtractor(sampling_rate=24000).save_pretrained(fixture/'tokenizer_saved')
 model.save_pretrained(fixture/'talker_saved')
else:
 loaded=Qwen3TTSTalkerModel.from_pretrained(fixture/'talker_saved',local_files_only=True)
 print('Saved talker load passed',sum(p.numel() for p in loaded.parameters()))
 loadedtok=Qwen3TTSTokenizer.from_pretrained(str(fixture/'tokenizer_saved'),local_files_only=True)
 print('Saved tokenizer load passed',sum(p.numel() for p in loadedtok.model.parameters()))
 with torch.inference_mode():
  restored=loadedtok.model.decode(results['encode'].unsqueeze(0)).audio_values[0]
  torch.testing.assert_close(restored,results['decode'])
 print('Saved tokenizer decode matched')
