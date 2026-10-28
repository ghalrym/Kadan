# Text to speech

Hikari turns a script into a completed WAV using LFM3-Audio-2B. Users choose one of the model card's supported languages/locales and a female or male voice. The English choices are American English and British English. There are no Qwen named speakers, automatic-language, free-form delivery instruction, clone or voice-description modes in this integration.

The checkpoint may still need an approved download or loading, and the request must fit available memory and the model's remaining context. The completed audio is mono 24 kHz, currently limited to sixty seconds. Oversized output, incomplete generation and invalid speech produce an error rather than a truncated clip. These controls do not promise pronunciation fidelity or voice consistency; listening and quality validation remain necessary.

Example text is editable and does not show that speech has already been generated. The form and generated clips stay only in the open page and disappear on navigation or reload. Save required audio while available. Playback starts only after complete validated audio arrives.

**Cancel request** ignores late results; worker cleanup can finish later. Leaving the page stops its request. Retrying submits the current form again and can produce different audio.

The existing Whisper file-transcription feature is separate. Call utterance and speech-duration limits do not establish full-file recognition parity with Whisper.

See [Speech synthesis architecture](../../architecture/hikari/worker-tts.md) for execution and input limits.
