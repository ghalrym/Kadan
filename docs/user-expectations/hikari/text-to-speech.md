# Text to speech

Hikari turns a script into a completed WAV using a named CustomVoice speaker, for Kadan and for other apps through its API; Kadan Server has no page for trying it. Requests choose a speaker and can describe delivery. Instructions guide that speaker; they do not clone a recording or create a new voice. Clone and Describe modes are unavailable.

The script's language is detected automatically. Unsupported speakers, languages or oversized inputs produce an error.

The fixed Qwen3-TTS CustomVoice checkpoint must already be downloaded and can still need loading, and the request must fit available memory. Audio is returned only once it is complete and validated. Hikari keeps no copy of generated clips.

Cancelling a request ignores late results; cleanup can finish later. Retrying sends a new request and can produce different audio.

See [Speech synthesis architecture](../../architecture/hikari/worker-tts.md) for supported inputs and byte limits.
