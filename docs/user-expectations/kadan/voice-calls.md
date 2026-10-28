# Voice calls

The **Call** action on an agent's card opens a separate spoken conversation using Hikari's fixed LFM3-Audio-2B model for recognition, replies and speech. Its license must be reviewed and its checkpoint downloaded before starting. Calls have no tools, access to workspace resources, or saved transcript; the text conversation remains separate. End call or leaving the chat releases the microphone and call history.

Kadan reads the agent's saved persona and any additional selected persona when the call starts, including the same persona only once. The call keeps this snapshot until it ends; later edits and selection changes apply to the next call. If a persona is unavailable, in trash, or cannot be read from Tsuchi, the call reports the problem and does not start. Combined spoken instructions, including Kadan's brief conversation instructions, must fit within 8,000 characters; oversized personas cause an error instead of being cut off. Calls use spoken-conversation instructions with no tools, separate from the shared text-agent instructions. See [Personas](personas.md).

The microphone keeps listening during spoken replies. Speaking interrupts the previous reply, discards its remaining audio, and cancels its generation. Only fully played segments are retained as assistant context after an interruption. Mute disables microphone input and discards unfinished speech while allowing the reply to play. Each utterance is limited to twenty seconds; call recognition currently requests English from LFM3-Audio. HTTPS or localhost and browser microphone permission are required. Headphones reduce acoustic feedback; energy-based voice detection and browser echo cancellation can still mistake noise or loudspeaker audio for speech.

Replies use LFM3's generated voice rather than a named Qwen speaker. The default path buffers a completed reply before playback; an optional sentence-TTS candidate buffers individual known-script segments. This is not incremental PCM playback, and gaps or model loading delays are possible. Call context is bounded to recent turns.

The existing worker owner still enforces physical RAM and per-GPU limits for the participating audio model. End, disconnect, or error releases its call residency hint and cancels its work. Changing or losing the configured audio checkpoint ends the call; start again once it is available.

Applications using Hikari directly have separate [live voice call expectations](../hikari/voice-calls.md), including responsibility for playback acknowledgements and interruption controls.

The [LFM3 audio trial](../hikari/lfm3-audio-trial.md) documents optional execution candidates and measured limits. Calls do not establish parity with standalone file transcription, and the existing Whisper file-transcription path remains separate.
