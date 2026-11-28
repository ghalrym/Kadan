# Transcription worker (`speech`)

Transcribes audio, up to 30 seconds of mono 16-bit 16 kHz PCM WAV, with native
Qwen3-ASR 1.7B (`Qwen/Qwen3-ASR-1.7B`, Apache 2.0).

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

- `/v1/audio/transcriptions`.

## Engine

- Native Qwen3-ASR 1.7B (`qwen3-asr-1.7b`), matching the upstream `QwenLM/Qwen3-ASR` architecture:
  - Pure-math 128-bin Slaney log-Mel frontend (`kadan::stt::slaney_mel_filters`), requiring no external mel-filter data files.
  - 3x Conv2d downsampling frontend (`downsample_hidden_size=480`) over 100-frame chunks, `conv_out` projection, sinusoidal positional embeddings, 24 Pre-LN Transformer encoder layers (`d_model=1024`, 16 heads, block-diagonal window attention), and two-layer MLP projector (`1024 -> 2048`).
  - 28-layer Qwen3-1.7B causal text decoder (`hidden_size=2048`, 16 Q heads, 8 KV heads, `head_dim=128`, QK-RMSNorm, `rope_theta=1,000,000`, tied embeddings) with per-request KV caching and structured `language <Name><asr_text>` output parsing.
- The optional request `model` can only name `qwen3-asr-1.7b`; admission and preparation resolve the same fixed checkpoint.
- Preparation exports `tokenizer.json` from the verified checkpoint's `vocab.json` + `merges.txt` + `tokenizer_config.json` and loads safetensors shards directly without converting weights.

## Memory

- **Minimum:** 2 GiB RAM and 2.4 GiB VRAM.
- **Preferred:** checkpoint size plus host headroom in RAM; BF16 GPU weight bank plus runtime workspace in VRAM.
- **Several GPUs:** each GPU it uses needs its own 2.4 GiB floor.

## Scheduling

- Priority 3.

## Input preparation

The browser converts decodable audio to mono 16-bit PCM WAV at 16 kHz. It does not trim or split clips, and recording does not automatically stop at the native 30-second limit. `POST /v1/audio/transcriptions` accepts a WAV data URL rather than fetching remote audio references.

Preparation requires a verified local download before exporting tokenizer assets; it never starts a network download. Because Qwen3-ASR 1.7B outputs punctuated, formatted transcripts directly, the separate S1-mini formatting stage and `whisper_formatting` preference have been removed.

The browser keeps audio and results locally. Cancel ignores late results while server cleanup finishes; navigation also stops recording, releases microphone access and aborts pending requests. Clipboard failure does not change the transcription result.

See the user expectations for [speech to text](../../user-expectations/hikari/transcription.md).
