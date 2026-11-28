# Speech to text and formatting

Hikari transcribes recorded speech for Kadan's voice input and for other apps through its API; Kadan Server has no page for trying it. Recording in Kadan requires a supported secure browser context, such as HTTPS or localhost.

Transcription currently supports English clips up to 30 seconds. Long or unsupported inputs must fail rather than silently lose audio.

Hikari uses Whisper medium and the saved formatting preference. Optional formatting with S1-mini cleans up the recognized text. If formatting fails, the raw transcript is still returned with an error, and clients can ask for the raw text to compare. Both versions need review for meaning and accuracy. Cancelling ignores late results while cleanup finishes.

Whisper can reuse intermediate audio data within one transcription. That state is released when the request completes, fails or is cancelled; it is not reused for a later clip. Transcription still needs enough temporary memory, and reuse does not guarantee a fixed response time.

When memory is available, Hikari requests additional GPU space to retain Whisper’s working weights. Smaller granted budgets can still run transcription by streaming weights. Available memory and transfers affect speed; Hikari does not promise a fixed response time or substitute another model to make it fit.

See [Transcription architecture](../../architecture/hikari/worker-speech.md) for audio conversion, model selection and formatting fallback.
