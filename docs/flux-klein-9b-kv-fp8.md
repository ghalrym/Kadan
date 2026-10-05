# FLUX.2 klein 9B KV FP8

Settings downloads and selects `flux-klein-9b-kv-fp8` for the existing image generation and editing flow.

The original quantized transformer is [flux-2-klein-9b-kv-fp8.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8/tree/8430c03ee457ea86907ed31a009cd56280df003d) at `8430c03ee457ea86907ed31a009cd56280df003d`. Its reported weight size is approximately 9.82 decimal GB. This excludes required companion assets; the combined size is discovered from verified source manifests before downloading.

Matching [black-forest-labs/FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv/tree/a6dfb36eca3a3906eb2fd460795adfb844e5fcce) at `a6dfb36eca3a3906eb2fd460795adfb844e5fcce` supplies `model_index.json`, transformer configuration, tokenizer, scheduler, text encoder and VAE. The BF16 transformer weights and weight index are excluded. Both source identities and pins are recorded in the completed bundle. The primary repository supplies its LICENSE license file.

The runtime uses `Flux2KleinKVPipeline` with 4 steps and no guidance argument (KV pipeline). The explicitly selected local FP8 file is decoded into dense BF16 on CUDA or FP32 on CPU and injected as the pipeline transformer. This is a dense fallback, not an optimized FP8/NVFP4 kernel path. Kadan reserves decoded memory and owns the pipeline lifecycle.

License: `flux-non-commercial-license`. FLUX.2 klein 9B KV FP8 has a non-commercial license and usage conditions. Commercial use requires separate rights; continuing does not grant them or approve Hugging Face access.

Validation uses synthetic safetensors and companion assets through metadata, integrity checks, atomic publication, persisted selection and native pipeline construction. No real weights, artifact tensor headers, GPU inference, model quality or performance were tested. Unsupported quantization payloads fail closed; actual published payload compatibility still needs a separately authorized model validation run. 
