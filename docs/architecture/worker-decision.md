# Decisions worker (`decision`)

Answers structured decisions (choices, scores and Noul answers) with the pinned
Laya model. It is small and CPU-only so agents get fast answers without taking
GPU memory from generation.

## Program and container

- Its own program, `kadan-worker-decision`, built from
  `services/hikari/inference_worker/workers/decision/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/decision/Dockerfile`: no CUDA.
- Its own Compose service, `hikari-worker-decision` (CPU only). It can be scaled, and
  run on another computer with `compose.worker.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `/v1/decisions`.

## Engine

- Laya on the CPU, inside a fixed 3 GiB host envelope.

## Memory

- **Minimum and preferred:** the 3 GiB envelope in RAM. It uses no VRAM.

## Scheduling

- Priority 1, shared with the language model. Lower-priority workers leave its
  RAM minimum free while it is not loaded.
- Its budget never includes a GPU.

## Constraints

- Keep it host-only. Giving it VRAM would compete with the language model for
  no benefit.
