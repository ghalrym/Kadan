# Image worker (`image`)

Generates and edits images with FLUX.2 Klein 4B Distilled (`flux.2-klein-4b`, Apache-2.0).

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

- Components (Qwen3-4B text encoder, Flux2Transformer2DModel denoiser, AutoencoderKLFlux2 VAE)
  are staged through VRAM one at a time. The engine therefore runs on far less VRAM
  than its ~15.98 GB checkpoint size.
- Compatible adjacent requests are prepared together as one batch. The batch
  size comes from free memory bounded by the worker's budget, never from raw
  free VRAM.

## Memory

- **Minimum:** 16 GiB RAM, the engine's own floor, and 2.4 GiB VRAM.
- **Preferred:** checkpoint size plus those floors.
- **Several GPUs:** each GPU it uses needs its own 2.4 GiB floor.

## Scheduling

- Priority 4. It leaves the language worker's minimum free while that worker is not
  loaded.
- When idle, it shrinks to its minimum so the language model can grow. Any
  budget change closes the engine and drops its learned batch sizes.

## Constraints

- Every probe the engine uses for sizing must be bounded by its budget.
  Otherwise it competes with workers on the same GPU.

## Generation and editing contracts

Image generation and editing share worker scheduling and admission so neither path bypasses resource limits. The browser starts with a square aspect, four images and a random seed. Prompts must be nonblank and at most 8,000 characters; supported counts are one, two and four, with aspects 1:1 (1024×1024), 4:3 (1152×864), 3:4 (864×1152) and 16:9 (1344×768). FLUX.2 Klein 4B Distilled runs its 4-step empirical-mu flow-matching Euler schedule (`steps` defaults to 4, CFG 1.0).

The browser normalizes PNG, JPEG and WebP uploads to PNG on a 32-pixel grid, with at most 3,072 pixels per side and 4,718,592 pixels total. Each upload is limited to 20 MiB. Masks must be opaque and match the original source dimensions; source and mask receive the same resize. API uploads must already be normalized PNG data URLs.

`POST /v1/images/edits` defaults to `mode: source` and `strength: 0.65`. Output follows source dimensions; independent width and height are rejected. Zero strength returns source pixels; full strength starts the editable area from noise. Intermediate strengths encode the source with `AutoencoderKLFlux2` (including patchify + batch-norm normalization) and use a suffix of its sampling schedule. Latent mask restoration and a final source-pixel composite preserve black mask regions, edit white regions and blend gray regions.

`mode: reference` uses native FLUX.2 multi-reference conditioning. Callers supply the primary `image` and optionally `reference_images` (1 to 3 additional PNG data URLs, up to 4 ordered conditioning references total). Each reference is VAE-encoded into patchified latent tokens assigned distinct 4D `(T, H, W, L)` rotary coordinates (`T = 10 * (k + 1)` for 0-based reference index `k`, while denoised target tokens sit at `T = 0`), matching `Flux2KleinPipeline`'s in-context image conditioning. Reference mode rejects a mask or an explicitly supplied strength. Each reference must contain at least 65,536 pixels. Output defaults to the primary reference size; independent `width` and `height` must be supplied together, on the same 32-pixel grid and within the side and total-pixel bounds.

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
**13.0**. The image worker's Dockerfile installs this pair by default:

```sh
docker build -f services/hikari/inference_worker/workers/image/Dockerfile -t hikari-worker-image services/hikari
```

For an existing LibTorch installation, use `-DCMAKE_PREFIX_PATH=<torch-prefix>`
and `-DKADAN_ENABLE_IMAGE_GENERATION=ON` alongside the full-worker CUDA build
options in `inference_worker/workers/image/Dockerfile`. CMake requires the pinned Torch package, rejects a
mismatched toolkit and fixes the architecture to SM86 without probing a GPU.
CPU unit/tool builds leave image generation disabled. Distribute the dependency
license notices with the binary; Docker retains the PyTorch and NVIDIA notices
under `/opt/kadan/share/kadan/licenses`. Hikari's license is unchanged.

### FLUX.2 Klein 4B ATen model stages

- **Text encoder (`Qwen3ForCausalLM`, 36 layers total, executed through layer 26):** Formats prompts with the Qwen3 `enable_thinking=False` chat template (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`), right-pads to fixed length 512 (`pad_token_id = 151643`), executes transformer layers `0..26` in BF16/FP32 with causal attention + `q_norm`/`k_norm`, and concatenates hidden states after layers `8, 17, 26` (`hidden_states[9, 18, 27]`) into `[B, 512, 7680]`.
- **Denoiser (`Flux2Transformer2DModel`, 5 double-stream + 20 single-stream blocks, `dim = 3072`, 24 heads × 128):** Uses shared `double_stream_modulation_img`, `double_stream_modulation_txt`, and `single_stream_modulation` (modulations kept in FP32 for normalization and `(1 + scale)` before BF16 cast), SwiGLU feed-forward blocks (`linear_in`/`linear_out`), 4D `(32, 32, 32, 32)` rotary embeddings (`theta = 2000`), and `Flux2TimestepGuidanceEmbeddings` (`sigma * 1000`, distilled without guidance embedding). Reference latent tokens are concatenated onto the image stream for attention and sliced off before the final `norm_out` / `proj_out` velocity prediction.
- **Scheduler (`FlowMatchEulerDiscreteScheduler`):** Empirical-mu exponential time shift (`compute_empirical_mu` over target image sequence length and step count) with FP32 Euler steps (`sigma_0 = 1.0` to `0.0`).
- **VAE (`AutoencoderKLFlux2`, 32 latent channels → 128 patchified channels):** 2D GroupNorm32 + SiLU + Conv2d encoder/decoder with 2D mid-block linear attention, 2×2 spatial patchify/unpatchify (16× total spatial downsampling), and `bn.running_mean` / `bn.running_var` normalization (skipping `bn.num_batches_tracked`). Full-resolution FP32 execution falls back to halo-tiled convolutional blocks when workspace does not fit.

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

`KADAN_ATEN_IMAGE_TIMING` lines on stderr report synchronized `load`, `generation`
and `unload` seconds per component and the automatically chosen device.
TF32 remains disabled for FP32 matrix multiplication and cuDNN operations during
ATen generation, as does BF16 reduced-precision reduction.

## Subscription generation branch

On this branch, generation can be routed before native submission to the [Codex subscription image backend](subscriptions.md#image-generation-backend-on-this-branch). Local generation, source/reference editing, pet stages, and the native image runtime retain the contracts above. The subscription backend does not enter worker scheduling, download checks or physical memory admission. UI integration and owner validation remain pending; this does not establish behavior on master.
