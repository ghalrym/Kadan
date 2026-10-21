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

See [Image generation](../../user-expectations/hikari/image-generation.md) and [Image editing](../../user-expectations/hikari/image-editing.md) for user expectations. The [ATen execution notes](../../../services/hikari/README.md#c-aten-image-execution) retain build, placement and validation details.
