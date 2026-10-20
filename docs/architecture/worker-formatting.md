# Transcript formatting worker (`formatting`)

Cleans up Whisper transcripts with S1-mini by Superwhisper into semi-formal
prose. It is optional per request and saved as a setting.

## Program and container

- Its own program, `kadan-worker-formatting`, built from
  `services/hikari/inference_worker/workers/formatting/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/formatting/Dockerfile`: no CUDA; OpenBLAS.
- Its own Compose service, `hikari-worker-formatting` (CPU only). It can be scaled, and
  run on another computer using the same services in `compose.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- Formatting of transcripts after the transcription worker, when enabled.

## Engine

- S1-mini on the CPU, with its weights resident as floats. English input up to
  1,000 tokens and output up to 1,024 tokens.

## Memory

- **Minimum and preferred:** twice the checkpoint size plus 1 GiB of RAM. It
  uses no VRAM.

## Scheduling

- Priority 3, shared with transcription. Its budget never includes a GPU.

## Constraints

- A formatter failure returns the raw transcript with an error. It never fails
  the transcription.
