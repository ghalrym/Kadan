# Native H3 FL2VA text-to-video with audio

This provider implements original H3 FL2VA text-to-video, 4–15 seconds, 24 fps,
768-pixel short edge, with its native soundtrack. It does not implement Ref2VA
reference inputs, hosted Context-IR, or the unpublished 2K regeneration model.
Negative prompts are rejected because the official checkpoint is CFG-distilled.

Dependencies: the H3 download/catalog PR supplies `h3-fl2va` and
`model_manager.get_checkpoint`; the shared video jobs PR supplies the video route,
queue, cancellation and output serving. The provider consumes that queue's
`VideoSpec` without modifying the LLM selection.

The worker uses SGLang's native original-checkpoint loader, pinned to
`f048d5aa4bc1bcad7fa2c60d067590d83d6dbe4a`, and model revision
`42ed227ee7df40d41602854ae760620d6eb651fe`. It uses the H3 bundle's tokenizer,
processor, encoder, transformer and both VAEs. The original bundle is not the
Diffusers root-format bundle. No checkpoint Python is executed.

`api/requirements-h3.txt` describes an optional isolated environment;
`KADAN_H3_PYTHON` selects its interpreter. Nothing installs automatically. The
worker runs offline with local checkpoint paths. SGLang local-mode schedulers are
children of Kadan's job process; no HTTP inference server is launched. Kadan holds
exclusive shared RAM/VRAM admission and an active reservation until process-group
cleanup, including cancellation/failure. Completed output remains owned by the
shared video job store. The provider reserves twice the checkpoint estimate plus
16 GiB host staging, and the full checkpoint estimate plus 16 GiB on one GPU.
`KADAN_H3_RAM_BYTES` and `KADAN_H3_VRAM_BYTES` allow an operator to supply
measured workload budgets for layerwise offload. The conservative defaults reject
24 GiB cards; no 3090-compatible budget has been measured. Accounting is admission,
not a hard allocation limit or measured peak guarantee. It rejects insufficient capacity rather than pooling
multiple GPUs into fictional combined VRAM.

The baseline uses 50 evaluations, video shift 12 and audio shift 3. Torch compilation
is disabled; DiT and encoder use layerwise offload. No 3090 performance claim is
made. Actual weights/GPU tests, output quality, timing, and peak memory measurements
have not been run. Contract tests use lightweight fixtures only.

## Acceleration follow-up

The [ModelTC Turbo recipes](https://github.com/ModelTC/Minimax-H3-Turbo/tree/02e26d591f7a04d5d1a074c9566d5dd4f22f6225)
require distinct adapters, not fewer steps applied to the base checkpoint. Their
FL2VA 544p 4-step v0.1 and 8-step v1.0 recipes use video/audio shifts 12/3;
768p v1.0 4/8-step recipes use 6/3. Adapter filenames, immutable Hub revisions,
LoRA alpha/scale and sampler must be pinned together before enabling them. These
adapters are not downloaded, registered or silently applied by this provider.

Sources: [official checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe),
[official SGLang cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3),
[SGLang native generator](https://github.com/sgl-project/sglang/blob/f048d5aa4bc1bcad7fa2c60d067590d83d6dbe4a/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py).
The model license excludes the US, EU, UK and South Korea, including personal use;
the download acknowledgement does not confer usage rights.
