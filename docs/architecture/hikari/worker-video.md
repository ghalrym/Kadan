# Video worker (`video`)

Generates video with audio using H3 FL2VA INT8 + Turbo: 480p or 768p, 16:9,
9:16 or 1:1, 24 fps, 1 to 15 seconds.

## Program and container

- Its own program, `kadan-worker-video`, built from
  `services/hikari/inference_worker/workers/video/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/video/Dockerfile`: CUDA and FFmpeg, the only image with FFmpeg.
- Its own Compose service, `hikari-worker-video` (GPU). It can be scaled, and
  run on another computer using the same services in `compose.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `/v1/videos/generations`. Omitted clip settings use fixed five-second, 480p defaults.

## Engine

- Streams execution stages. Spare RAM beyond the working stage becomes its
  weight cache.
- The finished MP4 returns inline in the result, because the worker may run on
  another machine. The API writes the file it serves.

## Host upload ownership

Resident weight staging owns its admitted host bytes and transfers that owner into the media session before enqueue. Activation inputs are copied into an admitted owner in cancellable one-MiB chunks before any upload is submitted. Successful synchronization or a confirmed cleanup drain releases the source and its host charge. The existing execution allowance covers the largest activation copy: denoiser scratch uses 380,684 bytes per token and the largest input uses 21,504, together below the 496,128-byte per-token bound. Fixed staging and metadata retain their separate envelope; no nominal RAM grant increases.

An unconfirmed drain retains the complete session, including device operations/context, host upload source, RAM/VRAM charges and worker activity, through process lifetime. It cannot authorize idle/park or reuse. The quarantine root is initialized before submission and adds no allocation on the failure path. This retention is deliberate: static destruction cannot establish that a device stopped reading host memory. Worker restart is the recovery boundary. This upload-source change does not establish safe ownership of caller-provided download destinations after a failed drain; that remains a separate host-boundary limitation.

## Memory

- **Minimum:** the RAM working stage for the largest clip the API accepts
  (768p, 15 seconds), at least 4 GiB, and 3 GiB of VRAM.
- **Preferred:** that stage plus twice the checkpoint size in RAM, so the whole
  model can be cached; checkpoint size plus 3 GiB of VRAM.
- **Several GPUs:** each GPU it uses needs its own 3 GiB floor.

## Scheduling

- Priority 5: the lowest. It is the first to be offloaded when it is idle.
- It leaves the language worker's minimum free while that worker is not loaded.

## Constraints

- Results never go through a shared filesystem.

## Job ownership and retention

`POST /v1/videos/generations` accepts server-owned work with HTTP 202. Browser submission lifetime is separate from job lifetime: aborting the wait cannot revoke a job already accepted. Clients must reconcile the job list after an uncertain submission before retrying. Job cancellation requests cleanup of queued or running work, and clients inspect the resulting state.

The API's VideoJobs index is process-local. It tracks Queued, Rendering, Done, Failed and Cancelled records, while the frontend polls pending work about every three seconds. Only Done with an available artifact provides playable output. Completed MP4 bytes are written atomically beneath `HIKARI_VIDEO_DIR`, falling back to `KADAN_VIDEO_DIR` and then the local default. A content lookup requires the indexed Done job, so retained files do not reconstruct old routes after restart.

Separating transferred worker output from API file serving supports workers on other machines, but does not make this index durable. The user expectations for [video generation](../../user-expectations/hikari/video-generation.md) describe the download behavior.

## Request defaults

Go defaults to five seconds at 480p, Plus to eight seconds at 480p, and Pro to five seconds at 768p. The browser starts at 16:9. The API seed defaults to 42; the browser does not expose it. Prompts must be nonblank and at most 8,000 characters. Negative prompts are unsupported. Explicit settings still pass memory admission rather than being silently reduced.
