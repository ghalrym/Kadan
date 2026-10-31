# Live voice calls

Hikari lets applications open a live spoken conversation directly, without a Kadan agent. Calls use the fixed LFM3 language, Whisper medium transcription and Qwen3-TTS CustomVoice speech models. All three must already be downloaded and support inference. Call readiness confirms the session can begin; models can still need loading, and requests can wait or fail because of available memory.

The client supplies the call instructions, captures and submits utterances, plays returned audio and reports playback progress. Hikari provides recognition, reply text and synthesized speech. It does not capture a microphone, detect when the user starts speaking or control the client's speakers. Browser permission, recording controls and silence detection belong to the client. Calls have no agent tools or automatic access to Tsuchi resources.

Each new turn begins by interrupting the previous turn, including before the first utterance. Hikari cancels the previous response and waits for cleanup before acknowledging the interruption. The client must stop and discard old audio locally and ignore results belonging to older turns. Each turn accepts one utterance; starting another requires a newer turn. Cancellation can take time and does not undo audio already heard.

Replies arrive as text and short, separately synthesized audio segments using the English Ryan voice. Each segment is generated before delivery, so playback can have gaps. The client must acknowledge segments in order, only after each has fully played. Hikari carries only acknowledged assistant segments into the next turn's context. Missing acknowledgements can make it forget replies the user heard; acknowledging unplayed audio can make it assume the user heard words they did not. Generation finishing does not mean playback has finished.

Calls retain bounded recent context only for that connection. Ending or disconnecting cancels pending transcription, text generation and speech synthesis and discards the session history. Reconnecting starts a new call, without a saved transcript or automatic replay. Invalid input, invalid playback acknowledgements and inference failures report an error and close the call. Recognition and generated replies need review for accuracy; an utterance with no recognized text can finish without a reply.

During a call, its three models receive equal residency priority. This does not promise equal memory shares, simultaneous execution or that all three fit together. Ending, disconnection or error restores ordinary inference priorities and idle expiry.

See [Transcription](transcription.md) for supported audio and recognition limits, [Text to speech](text-to-speech.md) for synthesis limitations, and [Kadan voice calls](../kadan/voice-calls.md) for Kadan's microphone, mute and interruption controls.
