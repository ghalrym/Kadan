# Speech to text and formatting

Hikari lets users record audio with microphone permission or choose a file their browser can read. The clip can be previewed locally; **Transcribe** submits it. Recording requires a supported secure browser context, such as HTTPS or localhost.

Transcription currently supports English clips up to 30 seconds. Recording does not stop automatically at that limit. Long or unsupported inputs must fail rather than silently lose audio; choose a shorter clip when needed.

The page uses Whisper medium and the saved formatting preference. Optional formatting uses LFM to clean up the recognized text. If formatting fails, the raw transcript remains available with a visible error. **Raw transcription** lets users compare it with a formatted result. Both versions need review for meaning and accuracy.

**Discard** clears the clip and transcript. **Copy transcript** can fail independently of recognition. Retrying Copy clears the previous clipboard feedback while preserving the transcript and any recognition or formatting errors. Copy feedback applies only to the current attempt and result; choosing or discarding a clip or starting a new transcription clears it. Clips, results and form state disappear on navigation or reload.

**Cancel** ignores late results while cleanup finishes. Leaving the page also aborts pending work and releases microphone access.

See [Transcription architecture](../../architecture/hikari/worker-speech.md) for audio conversion, model selection and formatting fallback.
