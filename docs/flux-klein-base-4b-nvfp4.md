# FLUX.2 klein base 4B NVFP4

Settings downloads and selects `flux-klein-base-4b-nvfp4` for the existing image generation and editing flow.

The original quantized transformer is [flux-2-klein-base-4b-nvfp4.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-nvfp4/tree/e655535fb8c2c148b47308cfa559a86a73273d55) at `e655535fb8c2c148b47308cfa559a86a73273d55`. Its reported weight size is approximately 2.49 decimal GB. This excludes required companion assets; the combined size is discovered from verified source manifests before downloading.

Matching [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/a3b4f4849157f664bdbc776fd7453c2783562f4d) at `a3b4f4849157f664bdbc776fd7453c2783562f4d` supplies `model_index.json`, transformer configuration, tokenizer, scheduler, text encoder and VAE. The BF16 transformer weights and weight index are excluded. Both source identities and pins are recorded in the completed bundle. The primary repository supplies its LICENSE.md license file.

The runtime uses `Flux2KleinPipeline` with 50 steps and guidance 4.0. The explicitly selected local NVFP4 file is decoded into dense BF16 on CUDA or FP32 on CPU and injected as the pipeline transformer. This is a dense fallback, not an optimized FP8/NVFP4 kernel path. Kadan reserves decoded memory and owns the pipeline lifecycle.

License: `apache-2.0`. The checkpoint is distributed under Apache-2.0.

Validation uses synthetic safetensors and companion assets through metadata, integrity checks, atomic publication, persisted selection and native pipeline construction. No real weights, artifact tensor headers, GPU inference, model quality or performance were tested. Unsupported quantization payloads fail closed; actual published payload compatibility still needs a separately authorized model validation run. 
