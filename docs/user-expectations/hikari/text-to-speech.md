# Text to speech

Text to speech generates a completed WAV from a script using Qwen3-TTS 1.7B CustomVoice. The page selects the first supported model offered by the API and defaults to its Ryan speaker. The prefilled script and voice description are editable examples, not a live server-status report or evidence that a particular voice has been generated.

Choose a named speaker and optionally describe delivery in the voice-instruction field, then Generate speech. The browser uses Auto language. API clients can also select Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish or Italian. An instruction guides the selected speaker; it does not create a new voice or clone a recording. Describe and Clone models are not enabled native integrations and are not offered as working modes.

The script must contain nonblank text and fit 32,000 UTF-8 bytes. Voice instructions must fit 8,000 UTF-8 bytes. The displayed character count is not a byte count, so non-ASCII text can reach these limits sooner. Unsupported speakers, languages or settings produce an explicit error.

A model appearing in the choices identifies an enabled integration; it does not establish that its checkpoint is downloaded, loaded or able to fit the request. Acquisition, loading and execution failures must remain errors. Speech generation is buffered: the page receives and plays the complete WAV rather than incrementally playing native audio while it is synthesized.

Only validated audio responses become playable results. The current API returns an empty speech-history list and does not maintain a durable audio library. Generated clips and edits to the form remain in this page's memory and disappear on navigation or reload. Save required audio while it is available.

Cancel request aborts the browser request and ignores its late result; native cleanup can finish later. Leaving the page also aborts its request. Retrying submits the current form again and can produce different audio. The [inference admission and loading limits](inference.md) still apply.
