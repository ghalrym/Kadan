# S1-mini by Superwhisper

Native English transcript formatting uses the existing Transformers dependency, on CPU,
without an inference server or llama.cpp. The official checkpoint is
[`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini) at
`88f6b15896c73bbb13a3b596e0afe8ea0d5150b4`. The loader reads the local directory
`KADAN_MODEL_DIR/s1-mini-<revision>` (default model root `~/.local/share/kadan/models`),
or an explicit `KADAN_S1_MODEL_DIR`. Inference never downloads missing files.
The existing shared catalog/download integration is a separate dependency.

Retain the upstream LICENSE and NOTICE alongside any provisioned checkpoint.
The model's license requires the exact attribution **S1-mini by Superwhisper**.
[License](https://huggingface.co/superwhisper/s1-mini/blob/88f6b15896c73bbb13a3b596e0afe8ea0d5150b4/LICENSE)
and [NOTICE](https://huggingface.co/superwhisper/s1-mini/blob/88f6b15896c73bbb13a3b596e0afe8ea0d5150b4/NOTICE).
No weights are redistributed here.

Formatting uses the required system prompt and semi-formal/prose/general control line,
disables thinking, and decodes greedily. Long input is split at sentence boundaries
where possible, with each chunk capped at 1,000 tokenizer tokens. An oversized sentence
is split without dropping source characters. Independent chunks can lose cross-chunk
context. The raw transcript is always returned; empty normalized output is valid for
filler-only speech. Unsupported/unknown languages, busy requests and failures preserve
the raw text and expose an explicit formatting status.

Kadan reserves 6 GiB of host RAM before loading, including FP32 weights, checkpoint
mapping, tokenizer and bounded generation workspace. This conservative admission
estimate is not a measured RSS guarantee. No GPU is selected automatically. Loading,
inference and teardown run within one active reservation; no model remains resident
between requests. The transcription worker must finish this synchronous call before
releasing task ownership, even after caller cancellation.

Validation uses tiny tensor fixtures and fake models. Actual checkpoint inference,
quality, peak RSS and throughput have not been measured; no model weights were fetched.

The transcription endpoint applies formatting after Whisper returns and releases its
ASR resources. The existing Settings switch persists this browser's preference; each
recording submission sends it explicitly. API callers can set `formatting` directly.
The response exposes `raw_text`, `text`, `formatting_status` and `formatting_model`.
