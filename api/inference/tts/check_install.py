"""Check the patched Qwen runtime with tiny random CPU models, without weights.

Deferred imports keep these optional build checks out of normal API startup.
They exercise the actual upstream code and saved-model loading, including the
non-persistent rotary buffers that changed between Transformers versions.
"""


def check_speech_install():
    import torch
    import torchaudio
    resampled = torchaudio.functional.resample(torch.ones(1, 240), 24000, 16000)
    assert resampled.shape == (1, 160) and torch.isfinite(resampled).all()
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel
    config=Qwen3TTSTalkerConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,text_hidden_size=16,text_vocab_size=32,pad_token_id=None,rope_scaling={'rope_type':'default','mrope_section':[1,1,2],'interleaved':False},code_predictor_config={'vocab_size':32,'hidden_size':16,'intermediate_size':32,'num_hidden_layers':1,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':8})
    model=Qwen3TTSTalkerModel(config).eval()
    print('Tiny talker constructed')
    with torch.inference_mode():
     out=model(inputs_embeds=model.codec_embedding(torch.tensor([[1,2]])),use_cache=True)
     print('Tiny talker forward',out.last_hidden_state.shape)
     out2=model(inputs_embeds=model.codec_embedding(torch.tensor([[3]])),past_key_values=out.past_key_values,use_cache=True)
     print('Tiny talker cached forward',out2.last_hidden_state.shape,'cache',out2.past_key_values.get_seq_length())
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,Qwen3TTSTalkerForConditionalGeneration
    config.num_code_groups=3
    config.code_predictor_config.num_code_groups=3
    config.code_predictor_config.pad_token_id=None
    predictor=Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(config.code_predictor_config,config).eval()
    with torch.inference_mode():
     ids=predictor.generate(inputs_embeds=torch.randn(1,2,16),max_new_tokens=2,do_sample=False)
     print('Tiny predictor generate',ids.shape)
    talker=Qwen3TTSTalkerForConditionalGeneration(config).eval()
    with torch.inference_mode():
     generated=talker.generate(inputs_embeds=torch.randn(1,3,16),trailing_text_hidden=torch.randn(1,2,16),tts_pad_embed=torch.randn(1,1,16),max_new_tokens=3,do_sample=False,subtalker_dosample=False,return_dict_in_generate=True,output_hidden_states=True)
     print('Tiny talker generate',generated.sequences.shape)
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    encoder=dict(hidden_size=16,num_filters=8,num_residual_layers=1,upsampling_ratios=[2,2],num_hidden_layers=1,intermediate_size=32,num_attention_heads=2,num_key_value_heads=2,head_dim=8,codebook_size=16,codebook_dim=16,num_quantizers=2,num_semantic_quantizers=1,vector_quantization_hidden_dimension=16,upsample_groups=1,use_cache=False)
    decoder=dict(codebook_size=16,codebook_dim=16,hidden_size=16,latent_dim=16,num_attention_heads=2,num_key_value_heads=2,intermediate_size=32,num_hidden_layers=1,num_quantizers=2,upsample_rates=[2],upsampling_ratios=[2,2],decoder_dim=16,sliding_window=8)
    tok=Qwen3TTSTokenizerV2Model(Qwen3TTSTokenizerV2Config(encoder_config=encoder,decoder_config=decoder,encoder_valid_num_quantizers=2,decode_upsample_rate=8,encode_downsample_rate=8)).eval()
    print('Tiny tokenizer constructed')
    with torch.inference_mode():
     encoded=tok.encode(torch.randn(1,64),torch.ones(1,64,dtype=torch.bool))
     print('Tiny tokenizer encode',encoded.audio_codes[0].shape)
     decoded=tok.decode(encoded.audio_codes[0].unsqueeze(0))
     print('Tiny tokenizer decode',decoded.audio_values[0].shape)
    from qwen_tts.core.models import Qwen3TTSConfig,Qwen3TTSForConditionalGeneration
    from transformers import StoppingCriteria,StoppingCriteriaList
    from copy import deepcopy
    config=deepcopy(config)
    config.vocab_size=1100
    config.spk_id={'ryan':5}
    config.codec_language_id={'english':6}
    for i,key in enumerate(['codec_eos_token_id','codec_think_id','codec_nothink_id','codec_think_bos_id','codec_think_eos_id','codec_pad_id','codec_bos_id'],7):setattr(config,key,i)
    fullcfg=Qwen3TTSConfig(talker_config=config.to_dict(),tts_model_type='custom_voice',tts_model_size='tiny',tokenizer_type='qwen3_tts_tokenizer_12hz',im_start_token_id=1,im_end_token_id=2,tts_pad_token_id=3,tts_bos_token_id=4,tts_eos_token_id=5)
    full=Qwen3TTSForConditionalGeneration(fullcfg).eval()
    class Stop(StoppingCriteria):
     def __init__(self):self.calls=0
     def __call__(self,input_ids,scores,**kwargs):
      self.calls+=1
      return torch.full((input_ids.shape[0],),self.calls>=2,device=input_ids.device,dtype=torch.bool)
    stop=Stop()
    with torch.inference_mode():
     codes,_=full.generate(input_ids=[torch.tensor([[1,2,3,4,5,6,7,8]])],languages=['english'],speakers=['ryan'],non_streaming_mode=True,max_new_tokens=8,do_sample=False,subtalker_dosample=False,stopping_criteria=StoppingCriteriaList([stop]))
     print('Full tiny generator cooperative stopping calls',stop.calls,'codes',[x.shape for x in codes])
     assert stop.calls==2

    assert out.last_hidden_state.shape == (1, 2, 16)
    assert out2.past_key_values.get_seq_length() == 3
    assert ids.shape == (1, 2)
    assert generated.sequences.shape == (1, 3)
    assert encoded.audio_codes[0].shape == (8, 2)
    assert decoded.audio_values[0].shape == (64,)
    assert torch.isfinite(decoded.audio_values[0]).all()
    import tempfile
    from transformers import Wav2Vec2FeatureExtractor
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
    with tempfile.TemporaryDirectory() as folder:
     model.save_pretrained(folder)
     restored_model=Qwen3TTSTalkerModel.from_pretrained(folder,local_files_only=True,attn_implementation=model.config._attn_implementation)
     with torch.inference_mode():
      restored=restored_model(inputs_embeds=restored_model.codec_embedding(torch.tensor([[1,2]])),use_cache=True)
     torch.testing.assert_close(restored.last_hidden_state,out.last_hidden_state)
    with tempfile.TemporaryDirectory() as folder:
     tok.save_pretrained(folder)
     Wav2Vec2FeatureExtractor(sampling_rate=24000).save_pretrained(folder)
     restored_tokenizer=Qwen3TTSTokenizer.from_pretrained(folder,local_files_only=True)
     with torch.inference_mode():
      restored_audio=restored_tokenizer.model.decode(encoded.audio_codes[0].unsqueeze(0)).audio_values[0]
     torch.testing.assert_close(restored_audio,decoded.audio_values[0])
    print('Saved model/tokenizer round trips matched; compatibility smoke passed')


if __name__ == '__main__':
    check_speech_install()
