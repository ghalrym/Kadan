# Speech synthesis worker (`tts`)

Turns text into speech with the tier's speech model (Qwen3-TTS 1.7B
CustomVoice): named speakers, supported languages and instructions.

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
