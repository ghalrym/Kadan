# FLUX.2 klein 4B

The single `black-forest-labs/FLUX.2-klein-4B` checkpoint is pinned to `e7b7dc27f91deacad38e78976d1f2b499d76a294` under Apache 2.0. The native Diffusers bundle is approximately 16 GB: Qwen3 text encoder, tokenizer, transformer, VAE and scheduler. The root transformer export is deliberately excluded because it duplicates `transformer/` weights.

Select the completed download in the existing Image generation Settings row. This choice persists independently of the LLM. API callers can override `model` on generation/edit requests. The native `Flux2KleinPipeline` is present in the already pinned Diffusers source; its distilled recipe uses four steps and guidance 1.0 for both text generation and image-conditioned editing. Kadan's shared image lifecycle owns CPU/GPU admission, offload, cancellation, failure cleanup and atomic PNG publication. No ComfyUI runtime or external server is used.

This PR depends on native Qwen Image infrastructure. It adds one checkpoint, not the base or 9B variants. Tests use fixture images/pipelines only; real weights, generation quality, memory peaks and throughput have not been tested.

Sources: [official model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [pinned files](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294).
