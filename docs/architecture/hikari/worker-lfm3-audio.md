# Experimental LFM3 audio worker

This optional worker runs the released BF16 [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B), pinned to `9d49701c372d6011d900f6abb40c6b296fb706dd`. Its manifest verifies all required model, codec, tokenizer, configuration and custom-code files. The worker uses the experimental transport and Hikari's existing explicit license acknowledgement and download endpoint. It never adds the model to a hardware tier or changes the native speech, transcription, image or video workers.

This is a user-directed dense-model experiment alongside the project's MoE language tiers. Native Hikari does not implement LFM3. The separate Python worker uses the model's pinned custom Transformers implementation, with local-only loading, BF16, one CUDA GPU, and the bundled LiquidCodec. No API process imports or runs the inference libraries.

`AudioCallTask` sends either a transcription request with one WAV utterance or a reply request with the call's text history. Both route to `lfm3_audio`, enter normal memory admission and execute in a child process. Cancellation terminates and joins that process before cleanup is acknowledged. By default the trial loads a fresh model for each queued request, so a turn requires two model loads. `HIKARI_LFM3_AUDIO_SINGLE_TURN=true` on the API selects an alternative `turn` task: one granted disposable child loads once, transcribes the utterance, and then generates the buffered reply with the same model, processor and codec. Each generation starts with a fresh cache; ASR state is not reused for the differently prompted reply. The child still exits and joins before memory release. Neither path retains an engine or prefix cache between turns.

The single-turn path emits a typed `audio_transcript` worker event before reply generation. Hikari forwards this through a bounded audio event queue, remembers the user text and delivers the ordinary transcript event immediately. A later reply failure or interruption preserves that recognized utterance. The worker and call controller use the same pure history-retention function when constructing the reply context, so fusing the operations does not change which prior messages are included. Reply text and PCM still wait for the existing EOS and privacy checks, and only a played, acknowledged reply enters assistant history. Cancelling the stream uses the existing cancel-and-join path; no new utterance is acknowledged before cleanup.

The adapter accepts only mono PCM16 16-kHz audio, at most twenty seconds. Its full released 32,768-token audio context reserves 1,024 generation steps, so prompts above 31,744 input tokens are rejected without truncation. The checkpoint exports a 128,000-position text backbone; that does not establish a larger supported multimodal context than the audio model card advertises. Replies require the mixed turn's EOS/im_end plus both text and speech. The text-end marker starts the audio tail and cannot complete a reply; transcription-only output can end at text-end. Raw decoded output is checked before special markers are removed: unfinished reasoning and tool markup are rejected, and any reasoning in a spoken reply rejects the entire waveform because thought-to-audio alignment is unverified. Completed reasoning in transcription-only output is removed; PCM16 WAV output is mono 24 kHz and limited to sixty seconds. The existing call history retention remains bounded independently of the model context. No extra adapter character or turn cap reduces that context, and the advertised streaming modes remain unexposed. The provisional grant is 8 GiB RAM plus 16 GiB VRAM, including model and codec. The short-utterance trial below measured one request; maximum-context fit and coexistence with other models remain unverified.


## Private streaming codec probe

With single-turn mode enabled, `HIKARI_LFM3_AUDIO_STREAM_PROBE=true` on the API opts into a private decode experiment. It defaults to false. The task carries a strict `codec_streaming_probe` boolean, enabled only for complete turns, and ASR is unchanged. The reply uses the pinned model's `audio_streamer` and its bundled `LiquidCodecStreamingCache`. Non-EOS frames are decoded privately in groups of twelve with one cache and `finalize=False`; pending frames and one empty final flush are processed only after actual turn EOS, complete public-text checks and valid offline audio. The streamer's `end()` merely records completion because the model also calls it after errors and output-cap exhaustion.

The probe holds at most sixty seconds of finite CPU audio and a bounded pending-frame group. It emits only numeric progress (`audio_chunk_decoded`) after the first privately decoded chunk, enabling a cancellation trial during generation. It emits no early text, audio or playback acknowledgement. All model and codec state remains inside the granted disposable child and cancellation still terminates and joins it.

The returned waveform is always the original offline decode. The probe reports first-frame and first-decoded-chunk times from probe construction immediately before reply generation, after reply input preparation. These exclude ASR, cold imports, model/processor loading and therefore are not utterance-to-playback latency. It also reports decoded chunk/sample counts and the maximum absolute float-sample difference against offline decode. Sample-count mismatch or nonfinite/oversized samples fail the probe. Numerical differences are diagnostic; no unmeasured equality tolerance or latency improvement is asserted. Streaming and offline decoding both run in this mode, so it can be slower. Manual measurements and interruption checks are required before changing its default or exposing any streamed playback.

The Hikari call start accepts `backend: "lfm3"` only with `HIKARI_ENABLE_LFM3_AUDIO=true`. Omitting it preserves the native call. Kadan forwards this explicit choice after resolving the same saved persona snapshot. A frontend built or started with `VITE_ENABLE_LFM3_AUDIO=true` offers **Try LFM3 call** beside **Call**. The existing utterance, interruption, playback and disconnect protocol is unchanged. The whole buffered reply is one acknowledged segment, and only acknowledged assistant text enters later history. There is no extra agent authority or tool access.

## Morning validation

Use `compose.yaml` with `compose.lfm3-audio.yaml`, `HIKARI_ENABLE_LFM3_AUDIO=true` and profile `lfm3-audio` in an isolated test stack assembled from the experimental worker changes; do not replace running containers. The overlay enables the API and frontend flags only when explicitly set, and keeps the audio checkpoint in its own persistent volume. Build the experimental worker image and enable the audio profile in that stack. Starting only the worker with `up --no-deps hikari-worker-lfm3-audio` leaves other containers intact, but the API and frontend also need the reviewed branch code and flags before the trial is usable; an old API image cannot provide this feature. The operator must choose when to recreate those test services. Check that its checkpoint appears alongside the existing models. Review the linked LFM Open License v1.0, explicitly confirm its download through the existing model download control, and wait for completed status. No license is accepted or model downloaded by this change's tests.

1. Start a native call to establish the existing path still works, then end it. Choose **Try LFM3 call** or send `{"type":"start","backend":"lfm3"}` to the existing Hikari call socket.
2. Speak a short English utterance. Inspect its transcript, reply text and complete buffered playback. Record load time, peak RAM/VRAM and perceived speech quality. Readiness alone does not prove inference works.
3. Interrupt while recognition or reply generation is pending. Confirm old audio stops, cleanup completes before the next turn and no stale audio appears. Interrupt partly played speech, then compare with a fully acknowledged reply to check retained context.
4. Exercise mute, disconnect and a new call; check history resets and existing image/video workers remain available. Inspect jobs after any uncertain result before retrying.
5. Compare the same utterance with `HIKARI_LFM3_AUDIO_SINGLE_TURN=false` and `true` in the isolated API configuration. Confirm the transcript arrives before the reply, and repeat interruption/failure during reply generation to check that user history remains. Record actual elapsed time and memory; removing a second startup does not establish a particular latency or usability.
6. Disable the trial flag or omit the backend choice and confirm ordinary calls remain selected. No containers or models need removal to stop the trial.

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

### Single-executor comparison

A follow-up of the same generated utterance measured 14.30 seconds on the
original two-executor path. ASR used 2.59 seconds importing the backend,
0.75 loading the model, 0.66 preparing the processor/codec and 1.42 in the
audio operation. Reply repeated that setup, then spent 2.20 seconds in its
audio operation. Initial PyTorch setup is included separately in total
executor time. This identified repeated setup as the main short-turn cost.

With the default-off single-turn flag enabled only in the isolated trial,
the correct transcript arrived at 6.57 seconds and the correct buffered text
and WAV at 8.83 seconds. One executor spent 1.43 seconds in ASR and 2.00 in
reply, sharing its 2.58-second backend import, 0.75-second model load and
0.66-second processor setup. Peak RSS was 4.54 GiB and CUDA allocation
3.84 GiB. This is one observed comparison, about 38% faster; sampling and
reply duration can change latency. It remains slow for live calls.

No microphone was opened or sound played overnight. A separate loopback-only
frontend passed HTTP and API-proxy checks, but the browser independently
blocked localhost navigation with ERR_BLOCKED_BY_CLIENT. No policy, extension,
certificate or network setting was changed to bypass that block. A saved
no-autoplay WAV player provides a user-driven morning listening option.

The owner-authorized private streaming probe decoded five finite chunks totalling
74,880 samples for one real reply. Its first generated audio frame arrived
0.232 seconds and its first private PCM chunk 0.683 seconds after probe
construction. The final validated WAV arrived at 8.83 seconds from call start.
Chunked and offline decoding had equal sample counts but were not bit-identical:
maximum absolute sample difference was 0.00724. Audible equivalence was not
validated, and the delivered waveform remained the original offline decode.

A separate real turn was interrupted after monitoring observed the first decoded
chunk. Cleanup was acknowledged in 0.215 seconds, no audio or late events
appeared in the bounded check, and GPU memory returned to idle. This establishes
generation/decoder interruption; microphone capture and audible-playback barge-in
remain unverified. The diagnostic flag was disabled after these measurements.
The official callbacks provide no reliable word-to-sample alignment, and their
end callback also runs after errors or output limits. The trial therefore keeps
speech behind final EOS/privacy validation and claims no earlier playback from
this diagnostic.
