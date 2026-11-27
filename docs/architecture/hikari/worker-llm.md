# Language worker (`llm`)

Serves public chat, agent tool-calling turns, and multimodal screenshot vision using Thinking Machines Lab's Inkling 2 Mini (`ggml-org/Inkling-2-Mini-GGUF`) on Hikari's native C++ Strata engine. This is the only local language model in Hikari; Qwen LLM checkpoints and the separate LFM3 text/vision Python workers have been removed.

## Program and container

- Its own program, `kadan-worker-llm`, built from
  `services/hikari/inference_worker/workers/llm/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/llm/Dockerfile`: CUDA; no LibTorch, no FFmpeg.
- Its own Compose service, `hikari-worker-llm` (GPU). It can be scaled, and
  run on another computer using the same services in `compose.yaml`.
- Configured via Compose environment variables:
  - `HIKARI_LLM_QUANT`: `IQ4_NL` (default) or `Q8_0`.
  - `HIKARI_LLM_ABLITERATED`: `false` (default, standard `ggml-org/Inkling-2-Mini-GGUF` checkpoints `inkling-2-mini-iq4-nl` / `inkling-2-mini-q8-0`) or `true` (abliterated `mrfakename/Inkling-2-Mini-abliterated-GGUF` checkpoints `inkling-2-mini-abliterated-iq4-nl` / `inkling-2-mini-abliterated-q8-0`).
  - `HIKARI_LLM_CONTEXT_LIMIT`: effective context window in tokens (default `65536`, range `[4096, 1048576]`).
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `POST /v1/chat/completions` (text, structured tools, and multimodal `image_url` screenshots) when the Language capability provider is set to `local`.
- Public lifecycle reads and explicit unload via `/model-lifecycle`.

## Engine

- Native GGUF looped-MoE architecture (`inkling2_loop`: 24 physical layers, 48 logical slots with 40 sliding-window and 8 full-attention slots) with a bundled vision projector (`mmproj`) for native image encoding.
- Supports Inkling 2 Mini's chat template, reasoning blocks, tool-call JSON and tool-role history, and multimodal image embeddings.
- **Single-GPU resident fast path:** When the packed weights and FP16 KV + F32 short-conv state fit on the single state GPU (e.g. `inkling-2-mini-iq4-nl` on a 24 GiB GPU), all 48 logical slots execute end-to-end in VRAM with zero host activation round-trips:
  - Packed `IQ4_NL` and `Q8_0` weight chunks are repacked on GPU at admission into 16B/32B-aligned SoA layout (`[contiguous qs] ++ [contiguous FP16 scales]`, same exact byte footprint, non-colliding `DeviceWeights::Key` widths) for single coalesced `LDG.32`/`LDG.64` loads.
  - Prompt prefill runs FP16 Tensor Core `cublasGemmEx` (`FP16 × FP16 -> FP32`) for dense projections (`m >= 16`) and async GPU-sorted CSR bucketing + grouped tiled quantized MMQ kernels (`gate_up + SiTU` and `down` across all 128 experts in 2 launches per layer, zero host syncs and zero temporary FP16 weight materialization) for MoE layers.
  - Token decode quantizes inputs to Q8 blocks and runs packed `IQ4_NL`/`Q8_0` `__dp4a` kernels with a fused multi-matrix projection megakernel (combining shared-input attention/control projections), fused Inkling kernels (`mix_states + rms_norm`, `ctrl_blend + rms_norm`, `qkvr_clip + tau + FP16 KV store`, `attn_fp16 + sigmoid gate`, `post_conv + residual`, indirect-GPU `moe_gate_up + SiTU` and deterministic `moe_down + reduce`, `output + softcap`), and 2-stage GPU `argmax` + finiteness checking (8-byte D2H result instead of 800 KB logits).
  - Single-token decode captures and replays all 48 slots plus the output head and `argmax` in a single CUDA Graph reading position from a 64-byte device parameter struct, automatically invalidating on KV mirror growth or request reset.
- **Strata tiered fallback:** When the model spans multiple GPUs or spills routed experts to host RAM/disk (e.g. `inkling-2-mini-q8-0` or constrained VRAM), execution falls back transparently to batched `GgufWeights::project` with specialized in-kernel `IQ4_NL`/`Q8_0` decoders, TF32 Tensor Core prefill, FP16 KV mirrors, and concurrent host `ggml-cpu` expert execution.
- Supports prefix cache reuse and a bounded multi-slot compacted checkpoint pool in host RAM (sized to one uncompacted KV/recurrent slot) holding per-conversation boundary and live checkpoints plus cross-conversation static system/tool prefix checkpoints, with image-span fencing so stale multimodal embeddings are never reused across mismatched images. On the single-GPU resident path, two bounded GPU-resident boundary snapshots (40 active FP16 SWA windows + 48 F32 short-conv states each, ~164 MiB per slot in `arena.recurrent`) allow warm multi-turn `checkpoint_hit` and `static_prefix_hit` rewinds to rewind the 8 append-only full-attention KV mirrors in place (0 bytes copied) and restore SWA/conv state device-to-device without re-uploading from host RAM.
- **Exact in-place variant hot-swap (base vs abliterated):** Standard (`inkling-2-mini-*`) and abliterated (`inkling-2-mini-abliterated-*`) Inkling 2 Mini checkpoints at the same precision (`IQ4_NL` or `Q8_0`) share 670 of 704 tensors (75.9% of bytes) bit-for-bit, differing only in the 16 `blk.8..23.attn_output.weight` and 18 `blk.6..23.ffn_down_exps.weight` matrices (~4.26 GB at `IQ4_NL`, ~7.76 GB at `Q8_0`). When a request selects the companion variant via `weight_variant` (`"base"` or `"abliterated"`) and its checkpoint is present locally, `InferenceOwner` routes through `ChatEngine::try_hot_swap` instead of tearing down and reloading the engine:
  - The 670 shared tensors, `mmproj` vision encoder, tokenizer, KV/recurrent arenas, and captured single-GPU decode CUDA Graph stay permanently resident.
  - On the single-GPU resident path, `InklingGpuRunner::swap_variant` writes the 34 differing SoA matrices directly into the existing GPU buffer addresses (`wo.qs`/`wo.d` and `down_ptrs[e]`), preserving CUDA Graph validity without recapture. It automatically selects one of three memory tiers (`select_inkling_swap_tier`) based on remaining granted VRAM and host RAM: (1) `gpu_dual_buffer` (both variants' repacked deltas resident in VRAM for D2D fast-swap when `spare_vram >= 2 * delta + 512 MiB`), (2) `pinned_host_ram` (both variants' repacked deltas cached in pinned host RAM for async H2D swap when `spare_host >= 2 * delta + 2 GiB`), or (3) `stream_repack` (zero persistent cache; streams and repacks directly from the target GGUF into the existing GPU buffers).
  - On the Strata tiered fallback (multi-GPU or host-RAM/disk expert spill), `GgufWeights::commit_variant_checkpoint` retains all 670 shared tensors in device and host caches, retires only the 34 differing tensors via `Placement::evict_tensor`, and routes subsequent lookups for those tensors to the target variant GGUF.
  - Prefix cache state (`Prefix`, `CheckpointPool`, `PrefixSnapshot`, and GPU boundary slots) is tagged with the active variant (`"base"` vs `"abliterated"`). Switching variants snapshots the outgoing live conversation into `CheckpointPool` under its variant tag and resets the live prefix, guaranteeing activations never mix across variants while allowing warm restores when switching back.
- Reports separate timing telemetry for weight hot-swap (`weight_swap_ms`), cache restore, image encoding, prompt prefill, reasoning, first visible token, and decode throughput.
## Memory

- **Minimum:** model weights + vision projector + FP16 KV mirror, F32 short-conv/parameter and GPU boundary snapshot device state + VRAM headroom, plus host RAM for staging and prefix checkpoint state.
- **Context:** sized from the configured `HIKARI_LLM_CONTEXT_LIMIT` (default `65536`) or an explicit request context limit within the checkpoint's supported maximum.
- **Several GPUs:** the budget may span GPUs where supported, with per-GPU headroom enforced. At the default `IQ4_NL` quant and `65536` context, FP16 KV + GPU boundary device state (~2.67 GiB) allows the entire model to pack cleanly onto a single 24 GiB GPU, leaving other GPUs free for image and audio workloads.

## Scheduling

- Priority 1, shared with decisions and live voice calls: the highest.
- Within queued language requests, `interactive` requests (chat turns) run ahead of `background` requests between turns, with a fairness bound (at most 3 consecutive interactive turns ahead of waiting background work) and no mid-generation preemption.
- While it is not loaded, other workers leave its minimum free where possible so chat can start beside an image (FLUX.2 Klein 4B) request.
- When idle lower-priority workers hold memory, it takes back the VRAM it lacks so GPU sharing between LLM and image generation works automatically without manual unloading.
- Any change to its budget closes the engine cleanly before releasing memory; the next request reloads within the new budget.

## Constraints

- Keep the decode path on batched `GgufWeights::project` with in-kernel
  dequantization. No F32 expansion of weights on that path.
- Never silently truncate chat history or substitute another model to force a fit.
