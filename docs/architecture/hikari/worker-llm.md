# Language worker (`llm`)

Retains the native engine for explicit internal checkpoint requests. Public chat uses the separate fixed LFM3 worker. This engine serves
mixture-of-experts checkpoints larger than the GPU by tiering routed experts
across VRAM, RAM and disk (the Strata/FreeToken approach).

## Program and container

- Its own program, `kadan-worker-llm`, built from
  `services/hikari/inference_worker/workers/llm/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/llm/Dockerfile`: CUDA; no LibTorch, no FFmpeg.
- Its own Compose service, `hikari-worker-llm` (GPU). It can be scaled, and
  run on another computer using the same services in `compose.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- Internal explicitly named native language requests. A native load may carry its own `context_limit`; public lifecycle controls belong to LFM.

## Engine

- Pinned GGUF mixture-of-experts checkpoints. The catalog retains their metadata without hardware profiles or saved selection; native language requests must name a checkpoint.
- Non-expert weights and context state stay pinned in VRAM. Only routed experts
  move between VRAM, RAM and disk.
- Routed experts held in RAM run on the CPU alongside the GPU batch. They are
  not streamed over PCIe by default.
- Prefill runs in 512-token chunks because each chunk touches nearly every
  routed expert. Experts stay on the batched path at any row count: the GPU
  repeats its packed pass for each group of 8 rows against weights it reads
  once, and the CPU split prices host work by bytes times rows.

## Memory

- **Minimum:** non-expert weights + context state + 2.4 GiB of VRAM headroom.
  RAM is the GGUF plan's host bytes + 4 GiB. Routed experts are the rank-3
  tensors with more than one expert.
- **Preferred:** every expert in VRAM, plus a RAM cache of the whole model.
- **Context:** sized from a queued load's explicit context first, then the
  saved limit, then the model's supported maximum.
- **Scratch:** each GPU keeps a 1 GiB scratch envelope for projections and
  grouped attention; the rest of the grant goes to the weight bank.
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
