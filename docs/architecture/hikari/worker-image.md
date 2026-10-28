# Image worker (`image`)

Generates and edits images with Qwen Image 2.1 Turbo.

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

- `/v1/images/generations` and `/v1/images/edits`.

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

## Generation and editing contracts

Image generation and editing share worker scheduling and admission so neither path bypasses resource limits. The browser starts with a square aspect, four images and a random seed. Prompts must be nonblank and at most 8,000 characters; supported counts are one, two and four, with aspects 1:1, 4:3, 3:4 and 16:9. The API's base step default is 50; Turbo uses its saved eight-step schedule.

The browser normalizes PNG, JPEG and WebP uploads to PNG on a 32-pixel grid, with at most 3,072 pixels per side and 4,718,592 pixels total. Each upload is limited to 20 MiB. Masks must be opaque and match the original source dimensions; source and mask receive the same resize. API uploads must already be normalized PNG data URLs.

`POST /v1/images/edits` defaults to `mode: source` and `strength: 0.65`. Output follows source dimensions; independent width and height are rejected. Zero strength returns source pixels; full strength starts the editable area from noise. Intermediate strengths encode the source with the checkpoint VAE and use a suffix of its sampling schedule. Latent mask restoration and a final source-pixel composite preserve black mask regions, edit white regions and blend gray regions.

`mode: reference` uses native reference conditioning for character and appearance. It rejects a mask or an explicitly supplied strength. The reference must contain at least 65,536 pixels. Output defaults to the reference size; independent `width` and `height` must be supplied together, on the same 32-pixel grid and within the side and total-pixel bounds. This is a separate conditioning contract from source denoising and pixel preservation.

## Cancellation and result ownership

API clients can supply `X-Kadan-Request-Id` as a UUID and use `POST /v1/images/requests/{request_id}/cancel`. A `cancelled` result joins active cleanup. `not_active` means no active request has that identity; it does not prove that work never ran or recover earlier output. Clients with durable storage must reconcile it before resubmitting uncertain work.

The playground aborts its browser request on cancellation or navigation and ignores late results. The image-history route returns an empty list; gallery bytes and form state live only in page memory. Kadan separately stores delivered images through Tsuchi. Keeping output ownership explicit prevents a browser gallery from being mistaken for durable storage.

See the user expectations for [image generation](../../user-expectations/hikari/image-generation.md) and [image editing](../../user-expectations/hikari/image-editing.md). The [ATen execution notes](#aten-image-execution) below retain build, placement and validation details.

## ATen image execution

C++ ATen is the supported image execution path in the existing serial worker.
Image generation requires CUDA; there is no CPU or custom resident image backend
fallback. API queueing, cancellation, image count, prompt validation, seed
progression and PNG publication remain owned by the same worker and memory ledger.
There is no Python inference executor or secondary service.

The supported dependency pair is PyTorch/LibTorch **2.14.1+cu130** and CUDA
**13.0**. The API Dockerfile installs this pair by default:

```sh
docker build -f services/hikari/api/Dockerfile -t hikari-api services/hikari
```

For an existing LibTorch installation, use `-DCMAKE_PREFIX_PATH=<torch-prefix>`
and `-DKADAN_ENABLE_IMAGE_GENERATION=ON` alongside the full-worker CUDA build
options in `api/Dockerfile`. CMake requires the pinned Torch package, rejects a
mismatched toolkit and fixes the architecture to SM86 without probing a GPU.
CPU unit/tool builds leave image generation disabled. Distribute the dependency
license notices with the binary; Docker retains the PyTorch and NVIDIA notices
under `/opt/kadan/share/kadan/licenses`. Hikari's license is unchanged.

The accepted October 13 fox comparison used build
`bdf012f034158c27b08ec13ff78c8cd39aa87a2a`, ATen enabled and TF32 disabled,
2048×2048 Turbo, eight Euler steps and CFG 1. Generation took **52.339790 s** in
Kadan and **53.031068 s** in ComfyUI. Kadan loading and unloading took 6.402629 s
and 0.580019 s separately. This was one sample per engine, with different noise
implementations; it establishes the accepted path for that workload, not broad
performance or numerical parity. The evidence is the October 13 task-26
`comparison.json`, `kadan-turbo.png` and `comfy-turbo.png`. Later arithmetic
corrections are documented below; the measured timings belong to the stated build.

The execution comparison uses official ComfyUI commit
[`8010b48d2c1be986613f8e8c761527d205bd756f`](https://github.com/Comfy-Org/ComfyUI/tree/8010b48d2c1be986613f8e8c761527d205bd756f),
specifically `comfy/text_encoders/qwen_image21.py` and
`comfy/ldm/qwen_image21/model.py`. This is an independently composed ATen model
implementation, not a direct ComfyUI source port. Its modulation keeps
normalization and `(1 + scale)` in FP32 until the final BF16 cast, following the
arithmetic boundary in Apache-2.0 Comfy Kitchen
[`be003b7c23c5b01328657955b8bc5d3f073d868e`, `comfy_kitchen/backends/cuda/ops/adaln.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/be003b7c23c5b01328657955b8bc5d3f073d868e/comfy_kitchen/backends/cuda/ops/adaln.cu)
(Copyright 2025 Comfy Org). Shared scales and tanh gates are prepared once per
step; residuals use ATen `addcmul_` with one output rounding. CPU arithmetic unit
fixtures verify these boundaries without a model or CUDA execution.

This composition does not reproduce Comfy Kitchen’s CUDA reduction tree or
kernel fusion. ComfyUI also fuses gate/up projections and forms rotary tables in
FP64 before casting; Hikari retains separate projections and FP32 rotary tables.
Matching those kernels, equal-seed noise, or full numerical output is not
established by this correction. No GPL source or upstream kernel source is
copied, and no license is added to Hikari.

Image stages reuse component weight banks retained on an admitted GPU when their
workspace fits. Each bank keeps a shared-memory-manager reservation, is pinned
while active, and can be evicted after synchronization when another stage needs
that device. Workspace is request-local. Model changes, competing workloads,
cancellation and execution failures release the image caches through the existing
worker owner. Physical cleanup is confirmed before releasing reservations.

Serialized checkpoint weights also remain in the existing bounded host cache,
using at most half the RAM left after request and stage staging budgets. GPU
uploads still occur on a device-cache miss. If whole-component residency does not
fit even after reclaiming inactive banks, the existing operator-streaming path
remains available. Weights keep their checkpoint precision.

Denoiser placement compares measured loading plus concurrent denoising time and
the reload cost of component banks it would displace. It starts conservatively,
permits one successful parallel calibration per comparable shape/cache regime,
and retains both-GPU execution when the measured cost or memory constraints favor
it. This is a bounded estimator, not a guarantee of optimal placement. In
particular, 24 GiB cards cannot keep both the approximately 14.1 GiB text bank
and 13.3 GiB denoiser bank on one device. Keeping them on separate GPUs can avoid
reloads that two simultaneous denoiser replicas would cause.

Text stores BF16 weights and performs FP32 forward operations; denoising keeps
BF16 weights/hidden states and uses library SDPA, normalization, SwiGLU and linear
operations. Text prefix K/V is cached for the current sampling request only. The
FP32 Euler latent and Hikari noise/seed contract are retained.

VAE computation stays on the GPU in FP32. If the full-resolution workspace does
not fit, the decoder keeps global attention on the full latent grid, then runs
its convolutional up blocks in 16-cell latent tiles with a 13-cell halo. Only
the tile interiors are published. The encoder tiles its convolutional down blocks
with a nine-cell halo and assembles the complete bottleneck before global attention.
Halos cover convolution dependencies and preserve stride alignment; no blending or
independent per-tile attention is used. The 1152-channel VAE attention still uses
256-query SDPA slices to bound scratch. Different convolution shapes may select
different library kernels; numerical parity and lower-tier performance require
manual validation on the installed model.

All stage plans are preflighted before the queue commits to running. Preparation
and execution use the same planner, and stage admission caps ATen's allocator.
Cancellation is checked during weight reads, layers, blocks and tile loops. If
neither resident nor streamed/tiled execution fits the available RAM and GPU
budgets, admission fails; there is no CPU execution fallback.

ATen caches CUDA handles for the process lifetime. In the full worker, all
workloads synchronize and free their owned allocations on retirement without
`cudaDeviceReset`, so switching workloads cannot invalidate ATen handle pools.
Those library/context pools survive until worker exit. Fresh physical memory
probes account for that baseline before the next engine is admitted. Component
weights are retained only under explicit cache reservations. Activations and
unused allocator blocks are released at stage boundaries; model retirement also
releases all cached banks before ledger acknowledgment. Failed cleanup retains
admission and blocks worker idle.
Repeated model switching remains a manual lifecycle check.

`KADAN_ATEN_IMAGE_TIMING` lines on stderr report synchronized `load`, `generation`
and `unload` seconds per component and the automatically chosen device. Sum the
`generation` records for the loading-free comparison (three for generation,
four when editing also encodes a source image). Transfers,
operator setup and conversion to owned CPU outputs are included; tokenizer/noise
preparation and PNG/API publication are outside these timers. The former resident
image timing option no longer drives worker execution; component timing utilities
remain available for unit checks and diagnostic tooling.

TF32 remains disabled for FP32 matrix multiplication and cuDNN operations during
ATen generation, as does BF16 reduced-precision reduction. This consolidation does not change the accepted
precision policy, sampling or model operators. New deployments and GPU validation
remain separate owner-coordinated actions.
