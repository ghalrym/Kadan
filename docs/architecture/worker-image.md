# Image worker (`image`)

Generates images with Qwen Image 2.1 Turbo. Editing is unavailable.

## Program and container

- Its own program, `kadan-worker-image`, built from
  `services/hikari/inference_worker/workers/image/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/image/Dockerfile`: CUDA and LibTorch, the only image with LibTorch.
- Its own Compose service, `hikari-worker-image` (GPU). It can be scaled, and
  run on another computer using the same services in `compose.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `/v1/images/generations`.

## Engine

- Components (text encoder, denoiser, VAE) are staged through VRAM one at a
  time. The engine therefore runs on far less VRAM than its checkpoint size.
- Compatible adjacent requests are prepared together as one batch. The batch
  size comes from free memory bounded by the worker's budget, never from raw
  free VRAM.

## Memory

- **Minimum:** 16 GiB RAM, the engine's own floor, and 2.4 GiB VRAM.
- **Preferred:** checkpoint size plus those floors.
- **Several GPUs:** each GPU it uses needs its own 2.4 GiB floor.

## Scheduling

- Priority 4. It leaves the language tier's minimum free while that tier is not
  loaded.
- When idle, it shrinks to its minimum so the language model can grow. Any
  budget change closes the engine and drops its learned batch sizes.

## Constraints

- Every probe the engine uses for sizing must be bounded by its budget.
  Otherwise it competes with workers on the same GPU.
