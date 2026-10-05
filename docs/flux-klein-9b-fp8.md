# FLUX.2 klein 9B FP8

Settings downloads and selects `flux-klein-9b-fp8` for the existing image generation and editing flow.

The original quantized transformer is [flux-2-klein-9b-fp8.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/tree/902d9d510b51533e07729f19211414a3648b77d2) at `902d9d510b51533e07729f19211414a3648b77d2`. Its reported weight size is approximately 9.43 decimal GB. This excludes required companion assets; the combined size is discovered from verified source manifests before downloading.

Matching [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/tree/92196c8e11f7b6cf2b7493e037d8c5345c559216) at `92196c8e11f7b6cf2b7493e037d8c5345c559216` supplies `model_index.json`, transformer configuration, tokenizer, scheduler, text encoder and VAE. The BF16 transformer weights and weight index are excluded. Both source identities and pins are recorded in the completed bundle. The primary repository supplies its LICENSE.md license file.

The runtime uses `Flux2KleinPipeline` with 4 steps and guidance 1.0. The explicitly selected local FP8 file is decoded into dense BF16 on CUDA or FP32 on CPU and injected as the pipeline transformer. This is a dense fallback, not an optimized FP8/NVFP4 kernel path. Kadan reserves decoded memory and owns the pipeline lifecycle.

License: `flux-non-commercial-license`. FLUX.2 klein 9B FP8 has a non-commercial license and usage conditions. Commercial use requires separate rights; continuing does not grant them or approve Hugging Face access.

Validation uses synthetic safetensors and companion assets through metadata, integrity checks, atomic publication, persisted selection and native pipeline construction. No real weights, artifact tensor headers, GPU inference, model quality or performance were tested. Unsupported quantization payloads fail closed; actual published payload compatibility still needs a separately authorized model validation run. The gated revision was resolved through the official public README commit hyperlink; authenticated pinned content was not fetched.
