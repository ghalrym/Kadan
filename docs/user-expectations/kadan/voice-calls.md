# Voice calls

The **Call** action on an agent's card opens a separate spoken conversation using the selected Hikari tier's language, transcription and speech models. All three must already be downloaded and available. The selected persona applies when the call starts. Calls have no tools, access to workspace resources, or saved transcript; the text conversation remains separate. End call or leaving the chat releases the microphone and call history.

The microphone keeps listening during spoken replies. Speaking interrupts the previous reply, discards its remaining audio, and cancels its generation. Only fully played segments are retained as assistant context after an interruption. Mute disables microphone input and discards unfinished speech while allowing the reply to play. Each utterance is limited to twenty seconds; recognition currently uses English Whisper. HTTPS or localhost and browser microphone permission are required. Headphones reduce acoustic feedback; energy-based voice detection and browser echo cancellation can still mistake noise or loudspeaker audio for speech.

Replies arrive as short Qwen3-TTS 1.7B CustomVoice audio segments with the Ryan speaker. Each segment is synthesized before it plays; this is not incremental PCM synthesis, and gaps or model loading delays are possible. Drama 3 is deferred. Call context is bounded to recent turns.

During a call, the three participating models have equal residency priority, without equal VRAM shares or a promise that they fit together. The existing worker architecture still enforces physical RAM and per-GPU limits. End, disconnect, or error restores normal priorities and idle expiry. Changing the selected tier ends the call; start again to use the new tier.

Applications using Hikari directly have separate [live voice call expectations](../hikari/voice-calls.md), including responsibility for playback acknowledgements and interruption controls.
