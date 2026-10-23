# Transcription worker (`speech`)

Transcribes English audio, up to 30 seconds of mono 16-bit 16 kHz PCM WAV, with
native Whisper.

## Program and container

- Its own program, `kadan-worker-speech`, built from
  `services/hikari/inference_worker/workers/speech/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/speech/Dockerfile`: CUDA and OpenBLAS.
- Its own Compose service, `hikari-worker-speech` (GPU). It can be scaled, and
  run on another computer using the same services in `compose.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `/v1/audio/transcriptions`. When formatting is on, the result then goes to the
  formatting worker.

## Engine

- Native Whisper at the size chosen in settings, or named by the request.
  Requests use sizes such as `medium` or `large`. The catalog stores them as
  `whisper-medium` and `whisper-large-v3`, so estimates and preparation
  normalize names the same way.

## Memory

- **Minimum:** 1 GiB RAM and 2.4 GiB VRAM.
- **Preferred:** checkpoint size plus those floors.
- **Several GPUs:** each GPU it uses needs its own 2.4 GiB floor.

## Scheduling

- Priority 3, shared with formatting.

## Constraints

- Name normalization must match `prepare_input(TranscriptionRequest&)`.
  Otherwise memory planning rejects valid requests.

## Input preparation and settings

The browser converts decodable audio to mono 16-bit PCM WAV at 16 kHz. It does not trim or split clips, and recording does not automatically stop at the native 30-second limit. `POST /v1/audio/transcriptions` accepts a WAV data URL rather than fetching remote audio references.

The page follows the saved Whisper checkpoint, defaulting to medium. API clients can save another supported selection or override `model` per request. Selection does not implicitly download a checkpoint. The catalog normalization above keeps preparation and admission consistent.

Formatting defaults on unless its saved setting changes. An omitted or null `formatting` field uses that setting; a boolean overrides it for one request. Successful recognition retains `raw_text` alongside `text`, with `formatting_status` equal to `disabled`, `formatted`, `failed` or `empty`, plus formatter identity/error when applicable. Empty raw input skips formatting. A formatter error returns the raw transcript instead of discarding successful recognition. See [Formatting worker](worker-formatting.md) for its input/output budgets and memory ownership.

The browser keeps audio and results locally. Cancel ignores late results while server cleanup finishes; navigation also stops recording, releases microphone access and aborts pending requests. Clipboard failure does not change the transcription result.

See the privately provided expectations for speech to text and formatting.

## Pinned Whisper assets

Tokenizer ranks and mel filters come from the MIT-licensed `openai-whisper` 20250625 source distribution (SHA-256 `37a91a3921809d9f44748ffc73c0a55c9f366c85a3ef5c2ae0cc09540432eb96`). `whisper_languages.json` preserves its tokenizer language order. `WHISPER_LICENSE` contains the upstream license.

The C++ worker verifies `manifest.json`, builds its English token table, and extracts the mel arrays. No Python package is used at runtime.
