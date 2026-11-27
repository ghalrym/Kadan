# Experimental Python workers

This directory hosts Hikari's Python-backed inference workers that use the
shared worker WebSocket transport, queue, worker token and memory scheduler:

- `audio.py` (`hikari-worker-lfm3-audio`): LFM3 Audio 2B BF16 for live voice calls.

Language, screenshot vision, and single-pass decisions are served natively by
Inkling 2 Mini in `hikari-worker-llm` (the C++ Strata MoE engine in
`../inference_worker`), not by Python workers in this directory.

Each worker uses its own pinned catalog JSON and model volume. No model weights
ship in the container image; downloads require explicit license acknowledgement
in Hikari Settings. See [LFM3 audio call architecture](../../../docs/architecture/hikari/worker-lfm3-audio.md)
for details on the audio call worker.
