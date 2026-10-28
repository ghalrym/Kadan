# Text to speech

Hikari turns a script into a completed WAV using a named CustomVoice speaker. Users can choose a speaker and describe delivery. Instructions guide that speaker; they do not clone a recording or create a new voice. Clone and Describe modes are unavailable.

The browser starts with Ryan and detects the script's language automatically. Example text is editable and does not show that speech has already been generated. Unsupported speakers, languages or oversized inputs produce an error.

A model in the choices may still need downloading or loading, and the request must fit available memory. Errors remain visible. Playback starts when the complete validated audio is available.

The form and generated clips stay only in the open page and disappear on navigation or reload. Save required audio while available.

**Cancel request** ignores late results; cleanup can finish later. Leaving the page also stops its request. Retrying submits the current form again and can produce different audio.

See [Speech synthesis architecture](../../architecture/hikari/worker-tts.md) for supported inputs and byte limits.
