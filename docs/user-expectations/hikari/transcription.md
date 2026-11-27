# Speech to text

Hikari lets users record audio with microphone permission or choose a file their browser can read. The clip can be previewed locally; **Transcribe** submits it. Recording requires a supported secure browser context, such as HTTPS or localhost.

Transcription supports clips up to 30 seconds. Recording does not stop automatically at that limit. Long or unsupported inputs must fail rather than silently lose audio; choose a shorter clip when needed.

Hikari uses Qwen3-ASR 1.7B directly to produce punctuated, formatted transcripts without a separate post-processing step. Choosing a file here does not give Kadan agents a file-transcription tool. Transcripts should still be reviewed for meaning and accuracy.

**Discard** clears the clip and transcript. **Copy** can fail independently of recognition. If clipboard access is unavailable or fails, Hikari selects the displayed transcript when the browser supports it and explains how to copy it with the browser’s Copy command or keyboard shortcut. If selection also fails, Hikari asks users to select the transcript themselves. Selecting text does not show **Copied**; that confirmation appears only after clipboard access succeeds. Retrying Copy clears the previous clipboard feedback while preserving the transcript and any recognition errors. Copy feedback applies only to the current attempt and result; choosing or discarding a clip or starting a new transcription clears it. Clips, results and form state disappear on navigation or reload.

**Cancel** ignores late results while cleanup finishes. Leaving the page also aborts pending work and releases microphone access.

When memory is available, Hikari retains Qwen3-ASR 1.7B's weights in GPU memory for faster transcription. Smaller granted budgets can still run transcription by streaming weights. Available memory and transfers affect speed; Hikari does not promise a fixed response time or substitute another model to make it fit.

See [Transcription architecture](../../architecture/hikari/worker-speech.md) for audio conversion and model details.
