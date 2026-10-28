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

- `/v1/videos/generations`. Omitted clip settings use the tier's defaults.

## Engine

- Streams execution stages. Spare RAM beyond the working stage becomes its
  weight cache.
- The finished MP4 returns inline in the result, because the worker may run on
  another machine. The API writes the file it serves.

Resident refiner, attention and feed-forward admission failures emit their existing fallback phase immediately after session cleanup, before host execution continues. Flushed `h3_fallback` stderr records include the internal planner detail and, when a rejected capacity plan exists, per-device arena and physical/ledger shortfall bytes. They contain no prompt or output paths. The API's acknowledged `load <worker> with <GiB> GiB` log records the actual grant; a later resize is a separate allocation, not evidence of the earlier running grant. Fallback diagnostics do not change the admission decision, computation or cleanup ownership.

`KADAN_H3_WALL_TIMING=1` opts the video worker into `H3_WALL` stderr JSON records. It is off by default. Each denoiser update separately measures resident refiner/block/feed-forward/final metadata description, preparation (admission plus weight reads/uploads), and execution (including initial step-input uploads and the existing completion/close boundaries). Execution records include the selected devices, arena and stored-weight bytes, token count and planned row/query/key tile sizes. Before admission, `plan_observed` is false and plan fields remain zero or empty. At most 192 phase records are emitted per update; its final summary retains aggregate call counts and durations, reports dropped observations, and marks exception unwinding and the cancellation flag. Diagnostic failures cannot replace the operation's outcome.

These are host wall durations, not measured kernel GPU times. They include existing synchronization and scheduling delays and do not separate CPU time, file I/O, PCIe transfers, cuBLAS or custom kernels. No waits or CUDA events are added. Host fallback computation, initial denoiser setup, encoding and other model stages are outside these resident timing buckets. A cancellation summary describes this scope's exit, not proof of request-level cleanup; the existing owner remains responsible for terminal acknowledgement and resource release.

Preparation records also contain one fixed-size `prepare_nested` aggregate for admission, staging allocation (the resource reservation and zero-initialized raw buffer), checkpoint reads/cache copies including the unchanged-file check, required finite/scale validation, and upload through its existing completion boundary. `included_in_outer_prepare_wall` is true: these are subdivisions of the outer prepare duration, never additional request time. Bucket calls count entries, not filesystem syscalls or GPU launches. The same nested totals are retained in the update summary; there are no per-slice log records.

Nested counters distinguish started and uploaded slices, planned bytes, and caller-completed read, validated and uploaded bytes for INT8, BF16, FP32 and other storage. A read or upload that throws may have made partial internal progress that is not counted; no completion is inferred. INT8 has no added validation scan, so its validated-byte count remains zero. Cancellation preserves completed counters and elapsed time in the active bucket. The prepare function’s explicit abort handler is outside nested buckets; any cleanup internal to admission remains included in admission wall time. Local object destruction during exception unwinding can still be charged to the active bucket before the abort handler. Untimed metadata work, normal buffer destruction and measurement overhead can leave outer prepare time unattributed. Neither timing nor byte counts establish a cache hit, physical disk I/O, bandwidth or the cause of a slowdown.

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

## Job ownership and retention

`POST /v1/videos/generations` accepts server-owned work with HTTP 202. Browser submission lifetime is separate from job lifetime: aborting the wait cannot revoke a job already accepted. Clients must reconcile the job list after an uncertain submission before retrying. Job cancellation requests cleanup of queued or running work, and clients inspect the resulting state.

The API's VideoJobs index is process-local. It tracks Queued, Rendering, Done, Failed and Cancelled records, while the frontend polls pending work about every three seconds. Only Done with an available artifact provides playable output. Completed MP4 bytes are written atomically beneath `HIKARI_VIDEO_DIR`, falling back to `KADAN_VIDEO_DIR` and then the local default. A content lookup requires the indexed Done job, so retained files do not reconstruct old routes after restart.

Separating transferred worker output from API file serving supports workers on other machines, but does not make this index durable. The user expectations for [video generation](../../user-expectations/hikari/video-generation.md) describe the download behavior.

## Request defaults

Go defaults to five seconds at 480p, Plus to eight seconds at 480p, and Pro to five seconds at 768p. The browser starts at 16:9. The API seed defaults to 42; the browser does not expose it. Prompts must be nonblank and at most 8,000 characters. Negative prompts are unsupported. Explicit settings still pass memory admission rather than being silently reduced.
