# Speech to text

Hikari transcribes recorded speech for Kadan's voice input and for other apps through its API; Kadan Server has no page for trying it. Recording in Kadan requires a supported secure browser context, such as HTTPS or localhost.

Transcription supports clips up to 30 seconds. Long or unsupported inputs must fail rather than silently lose audio.

Hikari uses Qwen3-ASR 1.7B directly to produce punctuated, formatted transcripts without a separate cleanup step. Transcripts should still be reviewed for meaning and accuracy. Cancelling ignores late results while cleanup finishes.

When memory is available, Hikari retains Qwen3-ASR 1.7B's weights in GPU memory for faster transcription. Smaller granted budgets can still run transcription by streaming weights. Available memory and transfers affect speed; Hikari does not promise a fixed response time or substitute another model to make it fit.

See [Transcription architecture](../../architecture/hikari/worker-speech.md) for audio conversion and model details.
