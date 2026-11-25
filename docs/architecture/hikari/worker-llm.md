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
  - `HIKARI_LLM_QUANT`: `IQ4_NL` (default → `inkling-2-mini-iq4-nl`) or `Q8_0` (→ `inkling-2-mini-q8-0`).
  - `HIKARI_LLM_CONTEXT_LIMIT`: effective context window in tokens (default `65536`, range `[4096, 1048576]`).
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `POST /v1/chat/completions` (text, structured tools, and multimodal `image_url` screenshots) when the Language capability provider is set to `local`.
- Public lifecycle reads and explicit unload via `/model-lifecycle`.

## Engine

- Dense GGUF multimodal architecture (`inkling2`) with a bundled F16 vision projector (`mmproj`) for native image encoding.
- Supports Inkling 2 Mini's chat template, reasoning blocks, structured `<tool_call>` / `<tool_response>` history, and multimodal image embeddings.
- Uses quantized weight projections with in-kernel dequantization (no F32 weight expansion on the decode path).
- Supports prefix cache reuse and conversation checkpoints across turns in the same conversation, with host-side detachment on offload.
## Memory

- **Minimum:** model weights + vision projector + KV cache state + VRAM headroom, plus host RAM for staging and prefix checkpoint state.
- **Context:** sized from the configured `HIKARI_LLM_CONTEXT_LIMIT` (default `65536`) or an explicit request context limit within the checkpoint's supported maximum.
- **Several GPUs:** the budget may span GPUs where supported, with per-GPU headroom enforced.

## Scheduling

- Priority 1, shared with decisions and live voice calls: the highest.
- While it is not loaded, other workers leave its minimum free where possible so chat can start beside an image (Qwen-Image) or video (H3) request.
- When idle lower-priority workers hold memory, it takes back the VRAM it lacks so GPU sharing between LLM and image/video generation works automatically without manual unloading.
- Any change to its budget closes the engine cleanly before releasing memory; the next request reloads within the new budget.

## Constraints

- Keep the decode path on batched `GgufWeights::project` with in-kernel
  dequantization. No F32 expansion of weights on that path.
- Never silently truncate chat history or substitute another model to force a fit.
