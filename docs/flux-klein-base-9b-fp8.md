# FLUX.2 klein base 9B FP8

Settings downloads and selects `flux-klein-base-9b-fp8` for the existing image generation and editing flow.

The original quantized transformer is [flux-2-klein-base-9b-fp8.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/tree/9ecf2143d71542449960c5584340269c6d401449) at `9ecf2143d71542449960c5584340269c6d401449`. Its reported weight size is approximately 9.57 decimal GB. This excludes required companion assets; the combined size is discovered from verified source manifests before downloading.

Matching [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/32773329fbe7e81a90ef971740e8ba4b0364ecf3) at `32773329fbe7e81a90ef971740e8ba4b0364ecf3` supplies `model_index.json`, transformer configuration, tokenizer, scheduler, text encoder and VAE. The BF16 transformer weights and weight index are excluded. Both source identities and pins are recorded in the completed bundle. The primary repository supplies its LICENSE.md license file.

The runtime uses `Flux2KleinPipeline` with 50 steps and guidance 4.0. The explicitly selected local FP8 file is decoded into dense BF16 on CUDA or FP32 on CPU and injected as the pipeline transformer. This is a dense fallback, not an optimized FP8/NVFP4 kernel path. Kadan reserves decoded memory and owns the pipeline lifecycle.

License: `flux-non-commercial-license`. FLUX.2 klein base 9B FP8 has a non-commercial license and usage conditions. Commercial use requires separate rights; continuing does not grant them or approve Hugging Face access.

Validation uses synthetic safetensors and companion assets through metadata, integrity checks, atomic publication, persisted selection and native pipeline construction. No real weights, artifact tensor headers, GPU inference, model quality or performance were tested. Unsupported quantization payloads fail closed; actual published payload compatibility still needs a separately authorized model validation run. The gated revision was resolved through the official public README commit hyperlink; authenticated pinned content was not fetched.
