# FLUX.2 klein 4B NVFP4

Settings downloads and selects `flux-klein-4b-nvfp4` for the existing image generation and editing flow.

The original quantized transformer is [flux-2-klein-4b-nvfp4.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4/tree/1db2b2f776c24b76f1122e5f69ab1949fc620068) at `1db2b2f776c24b76f1122e5f69ab1949fc620068`. Its reported weight size is approximately 2.46 decimal GB. This excludes required companion assets; the combined size is discovered from verified source manifests before downloading.

Matching [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294) at `e7b7dc27f91deacad38e78976d1f2b499d76a294` supplies `model_index.json`, transformer configuration, tokenizer, scheduler, text encoder and VAE. The BF16 transformer weights and weight index are excluded. Both source identities and pins are recorded in the completed bundle. The primary repository supplies its LICENSE.md license file.

The runtime uses `Flux2KleinPipeline` with 4 steps and guidance 1.0. The explicitly selected local NVFP4 file is decoded into dense BF16 on CUDA or FP32 on CPU and injected as the pipeline transformer. This is a dense fallback, not an optimized FP8/NVFP4 kernel path. Kadan reserves decoded memory and owns the pipeline lifecycle.

License: `apache-2.0`. The checkpoint is distributed under Apache-2.0.

Validation uses synthetic safetensors and companion assets through metadata, integrity checks, atomic publication, persisted selection and native pipeline construction. No real weights, artifact tensor headers, GPU inference, model quality or performance were tested. Unsupported quantization payloads fail closed; actual published payload compatibility still needs a separately authorized model validation run. 
