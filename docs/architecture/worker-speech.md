# Transcription worker (`speech`)

Transcribes English audio, up to 30 seconds of mono 16-bit 16 kHz PCM WAV, with
native Whisper.

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
