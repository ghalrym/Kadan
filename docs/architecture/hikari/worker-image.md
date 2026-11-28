# Image worker (`image`)

Generates and edits images with FLUX.2 Klein 4B Distilled (`flux.2-klein-4b`, Apache-2.0).

## Program

- Built directly into the unified `kadan-worker` binary (`--workload=image`) and
  also available as standalone `kadan-worker-image` from
  `services/hikari/inference_worker/workers/image/`.
- Pure native C++20/CUDA using cuBLAS and custom CUDA kernels (`flux2_cuda.cu`),
  with zero LibTorch or cuDNN dependency.
- Connects to `/hikari/v1/workers` and takes memory only as the API grants it.
  Workers on one computer share its RAM and GPUs in the API's ledger.

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

See the user expectations for [image generation](../../user-expectations/hikari/image-generation.md) and [image editing](../../user-expectations/hikari/image-editing.md). The [native CUDA execution notes](#native-cuda-image-execution) below retain build, placement and validation details.

## Native CUDA image execution

Pure C++20/CUDA (`src/cuda/flux2_cuda.cu` + `src/models/image/image_aten.cpp`) is
the supported image execution path in the existing serial worker. Image
generation requires CUDA; there is no CPU fallback, and no LibTorch or cuDNN
dependency. API queueing, cancellation, image count, prompt validation, seed
progression and PNG publication remain owned by the same worker and memory
ledger. There is no Python inference executor or secondary service.

`KADAN_ENABLE_CUDA=ON` builds `kadan-worker` with all four GPU workloads (`llm`,
`speech`, `tts`, `image`) using any modern NVIDIA CUDA toolkit (`cudart` +
`cublas`). CPU unit/tool builds leave CUDA disabled.

### FLUX.2 Klein 4B native CUDA model stages

- **Text encoder (`Qwen3ForCausalLM`, 36 layers total, executed through layer 26):** Formats prompts with the Qwen3 `enable_thinking=False` chat template (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`), right-pads to fixed length 512 (`pad_token_id = 151643`), executes transformer layers `0..26` in FP32 with causal+active GQA FlashAttention + `q_norm`/`k_norm`, and concatenates hidden states after layers `8, 17, 26` (`hidden_states[9, 18, 27]`) into `[B, 512, 7680]`.
- **Denoiser (`Flux2Transformer2DModel`, 5 double-stream + 20 single-stream blocks, `dim = 3072`, 24 heads × 128):** Uses shared `double_stream_modulation_img`, `double_stream_modulation_txt`, and `single_stream_modulation` (modulations kept in FP32 for normalization and `(1 + scale)` before BF16 cast), SwiGLU feed-forward blocks (`linear_in`/`linear_out`), 4D `(32, 32, 32, 32)` rotary embeddings (`theta = 2000`), tiled FP32-softmax BF16 FlashAttention, and `Flux2TimestepGuidanceEmbeddings` (`sigma * 1000`, distilled without guidance embedding). Reference latent tokens are concatenated onto the image stream for attention and sliced off before the final `norm_out` / `proj_out` velocity prediction.
- **Scheduler (`FlowMatchEulerDiscreteScheduler`):** Empirical-mu exponential time shift (`compute_empirical_mu` over target image sequence length and step count) with FP32 Euler steps (`sigma_0 = 1.0` to `0.0`).
- **VAE (`AutoencoderKLFlux2`, 32 latent channels → 128 patchified channels):** 2D GroupNorm32 + SiLU + chunked im2col/cuBLAS Conv2d encoder/decoder with 2D mid-block tiled attention, 2×2 spatial patchify/unpatchify (16× total spatial downsampling), and `bn.running_mean` / `bn.running_var` normalization (skipping `bn.num_batches_tracked`). Full-resolution FP32 execution falls back to halo-tiled convolutional blocks when workspace does not fit.

Image stages reuse component weight banks retained on an admitted GPU when their
workspace fits. Each bank keeps a shared-memory-manager reservation, is pinned
while active, and can be evicted after synchronization when another stage needs
that device. Workspace is request-local. Model changes, competing workloads,
cancellation and execution failures release the image caches through the existing
worker owner. Exact byte-level tracking in `flux2_cuda.cu` confirms every
transient allocation is released before closing a stage or acknowledging model
retirement.

Serialized checkpoint weights also remain in the existing bounded host cache,
using at most half the RAM left after request and stage staging budgets. GPU
uploads still occur on a device-cache miss. If whole-component residency does not
fit even after reclaiming inactive banks, the operator-streaming path remains
available. Weights keep their checkpoint precision.

All stage plans are preflighted before the queue commits to running. Preparation
and execution use the same planner. Cancellation is checked during weight reads,
layers, blocks, attention chunks and tile loops. If neither resident nor
streamed/tiled execution fits the available RAM and GPU budgets, admission
fails; there is no CPU execution fallback.

`KADAN_ATEN_IMAGE_TIMING` lines on stderr report synchronized `load`, `generation`
and `unload` seconds per component and the automatically chosen device.
cuBLAS uses `CUBLAS_PEDANTIC_MATH` (`CUBLAS_COMPUTE_32F_PEDANTIC`), keeping TF32
and reduced-precision BF16 reduction disabled.

## Subscription generation branch

On this branch, generation can be routed before native submission to the [Codex subscription image backend](subscriptions.md#image-generation-backend-on-this-branch). Local generation, source/reference editing, pet stages, and the native image runtime retain the contracts above. The subscription backend does not enter worker scheduling, download checks or physical memory admission. UI integration and owner validation remain pending; this does not establish behavior on master.
