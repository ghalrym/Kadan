# Speech to text and formatting

Hikari lets users record audio with microphone permission or choose a file their browser can read. The clip can be previewed locally; **Transcribe** submits it. Recording requires a supported secure browser context, such as HTTPS or localhost.

Transcription currently supports English clips up to 30 seconds. Recording does not stop automatically at that limit. Long or unsupported inputs must fail rather than silently lose audio; choose a shorter clip when needed.

Hikari uses Whisper medium and the saved formatting preference. This standalone transcription capability remains available independently of LFM3 audio calls; choosing a file here does not give Kadan agents a file-transcription tool. Optional formatting cleans up the recognized text. If formatting fails, the raw transcript remains available with a visible error. **Raw transcription** lets users compare it with a formatted result. Both versions need review for meaning and accuracy.

**Discard** clears the clip and transcript. **Copy transcript** can fail independently of recognition. If clipboard access is unavailable or fails, Hikari selects the displayed transcript when the browser supports it and explains how to copy it with the browser’s Copy command or keyboard shortcut. If selection also fails, Hikari asks users to select the transcript themselves. Selecting text does not show **Copied**; that confirmation appears only after clipboard access succeeds. Retrying Copy clears the previous clipboard feedback while preserving the transcript and any recognition or formatting errors. Copy feedback applies only to the current attempt and result; choosing or discarding a clip or starting a new transcription clears it. Clips, results and form state disappear on navigation or reload.

**Cancel** ignores late results while cleanup finishes. Leaving the page also aborts pending work and releases microphone access.

See [Transcription architecture](../../architecture/hikari/worker-speech.md) for audio conversion, model selection and formatting fallback.
