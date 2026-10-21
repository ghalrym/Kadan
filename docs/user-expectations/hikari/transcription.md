# Speech to text and formatting

Speech to text starts without a clip. Users can record with browser permission or choose an audio file their browser can decode, then preview it locally. Recording or choosing a file does not submit it automatically; press Transcribe. Microphone recording needs a browser-supported secure context such as HTTPS or localhost. Audio-file selection does not require microphone permission.

The browser converts the chosen clip to mono 16-bit PCM WAV at 16 kHz before submission. Native transcription accepts at most 30 seconds and currently supports English. The page does not automatically stop recording at that limit, trim a long clip or split it into several requests. Stop in time or choose a shorter clip; an overlong or unsupported input must fail rather than silently transcribing a truncated version.

Whisper uses its saved checkpoint selection, defaulting to medium unless changed. API clients can override the model per request or save another supported selection. Choosing a model does not download it implicitly. The page follows that saved selection rather than exposing its own model picker.

Formatting is enabled by default unless its saved setting was changed. After Whisper, S1-mini by Superwhisper normalizes the English transcript using semi-formal prose and general context. It has a 1,000-token input limit and a 1,024-token output limit. The formatter needs its downloaded checkpoint and CPU memory admission. API clients can change the saved switch or override formatting for one request; the page uses the saved value.

The response retains raw_text alongside the displayed text and identifies whether formatting was disabled, completed, failed or skipped for empty input. A formatting failure returns the raw transcript with a visible error rather than discarding successful recognition. An empty recognized transcript is not fabricated speech, and a formatted transcript still needs review for meaning and accuracy. Expand Raw transcription to compare a formatted result.

Discard clears the local clip and transcript. Copy transcript uses the browser clipboard and can fail independently of transcription. Clips and results are page-local and disappear on navigation or reload. Cancel stops the browser request and ignores late results; leaving the page also releases microphone access and aborts pending work. Server cleanup can finish later.
