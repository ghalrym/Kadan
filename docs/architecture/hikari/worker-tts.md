# Speech synthesis worker (`tts`)

Turns text into speech with the tier's speech model (Qwen3-TTS 1.7B
CustomVoice): named speakers, supported languages and instructions.

## Program and container

- Its own program, `kadan-worker-tts`, built from
  `services/hikari/inference_worker/workers/tts/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/tts/Dockerfile`: CUDA.
- Its own Compose service, `hikari-worker-tts` (GPU). It can be scaled, and
  run on another computer using the same services in `compose.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `/v1/audio/speech`.

## Engine

- The tier's `speech` model, or the request's `model_id`.

## Memory

- **Minimum:** 2 GiB RAM, the engine's own floor, and 2.4 GiB VRAM: compute
  context plus admission headroom.
- **Preferred:** checkpoint size plus those floors, so the whole model stays
  resident.
- **Several GPUs:** each GPU it uses needs its own 2.4 GiB floor.

## Scheduling

- Priority 2. It yields to the language tier and displaces transcription, image
  and video when they are idle or queued.

## Constraints

- Its minimums mirror the engine's load checks. A smaller grant would leave the
  engine waiting on memory it was promised.

## Input and result contract

The enabled integration is `qwen-tts-1.7b-custom`. `POST /v1/audio/speech` accepts nonblank script text up to 32,000 UTF-8 bytes and voice instructions up to 8,000 UTF-8 bytes. Character counts do not establish these byte limits. The endpoint validates the named speaker and language before execution.

The browser chooses the first enabled model, defaults to Ryan and uses Auto language. API clients can also select Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish or Italian. Describe and Clone have no enabled native integration. Listing an integration does not establish checkpoint residency or memory fit.

Speech is buffered into a complete WAV rather than incrementally played during synthesis. The history route is empty, and the playground retains clips and form edits in page memory only. Browser cancellation or navigation aborts the request and ignores late results while native cleanup finishes. Retry submits the current form as another inference.

See [Text to speech](https://chatgpt.com/space/page_a5eb66cabb10819192e8cd2b4371f238) for user expectations.
