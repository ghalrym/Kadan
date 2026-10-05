# FLUX.2 klein base 9B NVFP4

Settings downloads and selects `flux-klein-base-9b-nvfp4` for the existing image generation and editing flow.

The original quantized transformer is [flux-2-klein-base-9b-nvfp4.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4/tree/e651daf0c5d128e5cf7ffeb2da28fca22a8d7467) at `e651daf0c5d128e5cf7ffeb2da28fca22a8d7467`. Its reported weight size is approximately 5.81 decimal GB. This excludes required companion assets; the combined size is discovered from verified source manifests before downloading.

Matching [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/32773329fbe7e81a90ef971740e8ba4b0364ecf3) at `32773329fbe7e81a90ef971740e8ba4b0364ecf3` supplies `model_index.json`, transformer configuration, tokenizer, scheduler, text encoder and VAE. The BF16 transformer weights and weight index are excluded. Both source identities and pins are recorded in the completed bundle. The primary repository supplies its LICENSE.md license file.

The runtime uses `Flux2KleinPipeline` with 50 steps and guidance 4.0. The explicitly selected local NVFP4 file is decoded into dense BF16 on CUDA or FP32 on CPU and injected as the pipeline transformer. This is a dense fallback, not an optimized FP8/NVFP4 kernel path. Kadan reserves decoded memory and owns the pipeline lifecycle.

License: `flux-non-commercial-license`. FLUX.2 klein base 9B NVFP4 has a non-commercial license and usage conditions. Commercial use requires separate rights; continuing does not grant them or approve Hugging Face access.

Validation uses synthetic safetensors and companion assets through metadata, integrity checks, atomic publication, persisted selection and native pipeline construction. No real weights, artifact tensor headers, GPU inference, model quality or performance were tested. Unsupported quantization payloads fail closed; actual published payload compatibility still needs a separately authorized model validation run. The gated revision was resolved through the official public README commit hyperlink; authenticated pinned content was not fetched.
