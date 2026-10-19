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
  run on another computer with `compose.worker.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `/v1/videos/generations`. Omitted clip settings use the tier's defaults.

## Engine

- Streams execution stages. Spare RAM beyond the working stage becomes its
  weight cache.
- The finished MP4 returns inline in the result, because the worker may run on
  another machine. The API writes the file it serves.

## Memory

- **Minimum:** the RAM working stage for the largest clip the API accepts
  (768p, 15 seconds), at least 4 GiB, and 3 GiB of VRAM.
- **Preferred:** that stage plus twice the checkpoint size in RAM, so the whole
  model can be cached; checkpoint size plus 3 GiB of VRAM.
- **Several GPUs:** each GPU it uses needs its own 3 GiB floor.

## Scheduling

- Priority 5: the lowest. It is the first to be offloaded when it is idle.
- It leaves the language tier's minimum free while that tier is not loaded.

## Constraints

- Results never go through a shared filesystem.
