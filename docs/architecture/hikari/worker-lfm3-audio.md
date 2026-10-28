# Experimental LFM3 audio worker

This optional worker runs the released BF16 [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B), pinned to `9d49701c372d6011d900f6abb40c6b296fb706dd`. Its manifest verifies all required model, codec, tokenizer, configuration and custom-code files. The worker uses the experimental transport and Hikari's existing explicit license acknowledgement and download endpoint. It never adds the model to a hardware tier or changes the native speech, transcription, image or video workers.

This is a user-directed dense-model experiment alongside the project's MoE language tiers. Native Hikari does not implement LFM3. The separate Python worker uses the model's pinned custom Transformers implementation, with local-only loading, BF16, one CUDA GPU, and the bundled LiquidCodec. No API process imports or runs the inference libraries.

`AudioCallTask` sends either a transcription request with one WAV utterance or a reply request with the call's text history. Both route to `lfm3_audio`, enter normal memory admission and execute in a child process. Cancellation terminates and joins that process before cleanup is acknowledged. The trial currently loads a fresh model for each queued request, so a turn can require two model loads. It retains no engine or prefix cache.

The adapter accepts only mono PCM16 16-kHz audio, at most twenty seconds. Its full released 32,768-token audio context reserves 1,024 generation steps, so prompts above 31,744 input tokens are rejected without truncation. The checkpoint exports a 128,000-position text backbone; that does not establish a larger supported multimodal context than the audio model card advertises. Replies require the mixed turn's EOS/im_end plus both text and speech. The text-end marker starts the audio tail and cannot complete a reply; transcription-only output can end at text-end. Raw decoded output is checked before special markers are removed: unfinished reasoning and tool markup are rejected, and any reasoning in a spoken reply rejects the entire waveform because thought-to-audio alignment is unverified. Completed reasoning in transcription-only output is removed; PCM16 WAV output is mono 24 kHz and limited to sixty seconds. The existing call history retention remains bounded independently of the model context. No extra adapter character or turn cap reduces that context, and the advertised streaming modes remain unexposed. The provisional grant is 8 GiB RAM plus 16 GiB VRAM, including model and codec. The short-utterance trial below measured one request; maximum-context fit and coexistence with other models remain unverified.

The Hikari call start accepts `backend: "lfm3"` only with `HIKARI_ENABLE_LFM3_AUDIO=true`. Omitting it preserves the native call. Kadan forwards this explicit choice after resolving the same saved persona snapshot. A frontend built or started with `VITE_ENABLE_LFM3_AUDIO=true` offers **Try LFM3 call** beside **Call**. The existing utterance, interruption, playback and disconnect protocol is unchanged. The whole buffered reply is one acknowledged segment, and only acknowledged assistant text enters later history. There is no extra agent authority or tool access.

## Morning validation

Use `compose.yaml` with `compose.lfm3-audio.yaml`, `HIKARI_ENABLE_LFM3_AUDIO=true` and profile `lfm3-audio` in an isolated test stack assembled from the experimental worker changes; do not replace running containers. The overlay enables the API and frontend flags only when explicitly set, and keeps the audio checkpoint in its own persistent volume. Build the experimental worker image and enable the audio profile in that stack. Starting only the worker with `up --no-deps hikari-worker-lfm3-audio` leaves other containers intact, but the API and frontend also need the reviewed branch code and flags before the trial is usable; an old API image cannot provide this feature. The operator must choose when to recreate those test services. Check that its checkpoint appears alongside the existing models. Review the linked LFM Open License v1.0, explicitly confirm its download through the existing model download control, and wait for completed status. No license is accepted or model downloaded by this change's tests.

1. Start a native call to establish the existing path still works, then end it. Choose **Try LFM3 call** or send `{"type":"start","backend":"lfm3"}` to the existing Hikari call socket.
2. Speak a short English utterance. Inspect its transcript, reply text and complete buffered playback. Record load time, peak RAM/VRAM and perceived speech quality. Readiness alone does not prove inference works.
3. Interrupt while recognition or reply generation is pending. Confirm old audio stops, cleanup completes before the next turn and no stale audio appears. Interrupt partly played speech, then compare with a fully acknowledged reply to check retained context.
4. Exercise mute, disconnect and a new call; check history resets and existing image/video workers remain available. Inspect jobs after any uncertain result before retrying.
5. Disable the trial flag or omit the backend choice and confirm ordinary calls remain selected. No containers or models need removal to stop the trial.

Human microphone capture, audible playback quality, playback acknowledgements, and maximum-context fit still require operator validation. The adapter's unit tests use fake processors, models and queues and run no inference.

## Manual trial evidence

An owner-authorized trial used the actual Kadan call WebSocket relay and BF16
checkpoint with a clearly labelled generated English test utterance. Recognition
returned the correct transcript at 7.69 seconds; the correct arithmetic reply
and a complete mono 24-kHz PCM16 WAV arrived at 15.61 seconds. The waveform
contained about 3.12 seconds of speech. Recognition and reply each loaded a fresh
model: executor times were 7.39 and 7.61 seconds, peak RSS 4.54 GiB and CUDA
allocation/reservation about 3.84/3.85 GiB. Interrupting another pending
recognition turn was acknowledged in 0.106 seconds.

This latency is a limitation for live phone use. The LAN HTTP frontend rejected
microphone capture because calls require HTTPS or localhost. Browser playback
was blocked by the client, so no audible-quality or played-acknowledgement claim
is made and no played acknowledgement was sent. The test does not establish
full-duplex inference or interruption during audible playback. Native call
behavior was covered by unit tests, not a new live native-model call.
