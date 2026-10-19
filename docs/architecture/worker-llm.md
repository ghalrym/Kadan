# Language worker (`llm`)

Runs the hardware tier's language model for chat. It exists to serve
mixture-of-experts checkpoints larger than the GPU by tiering routed experts
across VRAM, RAM and disk (the Strata/FreeToken approach).

## Program and container

- Its own program, `kadan-worker-llm`, built from
  `services/hikari/inference_worker/workers/llm/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/llm/Dockerfile`: CUDA; no LibTorch, no FFmpeg.
- Its own Compose service, `hikari-worker-llm` (GPU). It can be scaled, and
  run on another computer with `compose.worker.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `/v1/chat/completions` (JSON and streaming).
- Model lifecycle `load` and `unload`. A load may carry its own `context_limit`.

## Engine

- One pinned GGUF mixture-of-experts checkpoint per hardware tier. Dense
  language models are never used.
- Non-expert weights and context state stay pinned in VRAM. Only routed experts
  move between VRAM, RAM and disk.
- Routed experts held in RAM run on the CPU alongside the GPU batch. They are
  not streamed over PCIe by default.

## Memory

- **Minimum:** non-expert weights + context state + 2.4 GiB of VRAM headroom.
  RAM is the GGUF plan's host bytes + 4 GiB. Routed experts are the rank-3
  tensors with more than one expert.
- **Preferred:** every expert in VRAM, plus a RAM cache of the whole model.
- **Context:** sized from a queued load's explicit context first, then the
  saved limit, then the model's supported maximum.
- **Several GPUs:** the budget may span GPUs. Each GPU it uses needs its own
  2.4 GiB floor.
- **Budget limits:** the GPU weight bank sizes itself from the smaller of free
  VRAM and the room left in its budget. Admission therefore never depends on the
  total number of expert slices.

## Scheduling

- Priority 1, shared with decisions: the highest.
- While it is not loaded, other workers leave its minimum free so chat can start
  beside an image or video.
- When idle lower-priority workers hold memory, it takes back the VRAM it lacks.
  They shrink to their minimum.
- Any change to its budget closes the engine. The next request reloads the whole
  model within the new budget; there is no live resize of the expert cache.

## Constraints

- Keep the decode path on batched `GgufWeights::project` with in-kernel
  dequantization. No F32 expansion of weights on that path.
- Never require the dense model to fit in VRAM to admit a request.
