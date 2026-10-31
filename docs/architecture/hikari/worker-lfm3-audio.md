# LFM3 audio call worker

The parent worker owns each executor's temporary directory. The child uses it as its Python temporary-file root, keeping ASR utterance WAVs inside that lifetime. After confirmed exit and join, the parent removes the directory before sending a terminal result, including when termination skipped the child's context-manager cleanup. Repeated cancellation still waits for the shielded teardown.

LFM3-Audio is the sole call backend in Hikari and Kadan. See [live voice call expectations](../../user-expectations/hikari/voice-calls.md). The worker is part of normal Compose startup; availability is not feature-flagged. Standalone Whisper/S1 transcription and Qwen synthesis retain their existing APIs, workers, settings and playgrounds. Kadan composer dictation still uses Whisper.

This worker runs the released BF16 [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B), pinned to `9d49701c372d6011d900f6abb40c6b296fb706dd`. Its manifest verifies all required model, codec, tokenizer, configuration and custom-code files. The worker uses the experimental transport and Hikari's existing explicit license acknowledgement and download endpoint. It never adds the model to a hardware tier or changes the native speech, transcription, image or video workers.

This dense-model worker runs alongside the fixed language/vision workers and retained native MoE path. Native Hikari does not implement LFM3. The separate Python worker uses the model's pinned custom Transformers implementation, with local-only loading, BF16, one CUDA GPU, and the bundled LiquidCodec. No API process imports or runs the inference libraries.

`AudioCallTask` sends either a transcription request with one WAV utterance or a reply request with the call's text history. Both route to `lfm3_audio`, enter normal memory admission and execute in a child process. Cancellation terminates and joins that process before cleanup is acknowledged. By default the worker loads a fresh model for each queued request, so a turn requires two model loads. `HIKARI_LFM3_AUDIO_SINGLE_TURN=true` on the API selects an alternative `turn` task: one granted disposable child loads once, transcribes the utterance, and then generates the buffered reply with the same model, processor and codec. Each generation starts with a fresh cache; ASR state is not reused for the differently prompted reply. The child still exits and joins before memory release. Neither path retains an engine or prefix cache between turns.

The single-turn path emits a typed `audio_transcript` worker event before reply generation. Hikari forwards this through a bounded audio event queue, remembers the user text and delivers the ordinary transcript event immediately. A later reply failure or interruption preserves that recognized utterance. The worker and call controller use the same pure history-retention function when constructing the reply context, so fusing the operations does not change which prior messages are included. Reply text and PCM still wait for the existing EOS and privacy checks, and only a played, acknowledged reply enters assistant history. Cancelling the stream uses the existing cancel-and-join path; no new utterance is acknowledged before cleanup.

The adapter accepts only mono PCM16 16-kHz audio, at most twenty seconds. Its full released 32,768-token audio context is shared by input and generated slots. Each operation permits every remaining slot after the exact tokenized prompt, including audio placeholders, instead of reserving a fixed 1,024-token reply. A prompt occupying the whole context fails because no reply slot remains; no retained input is cropped to fit. The checkpoint exports a 128,000-position text backbone; that does not establish a larger supported multimodal context than the audio model card advertises. Replies require the mixed turn's EOS/im_end plus both text and speech. The text-end marker starts the audio tail and cannot complete a reply; transcription-only output can end at text-end. Raw decoded output is checked before special markers are removed: unfinished reasoning and tool markup are rejected, and any reasoning in a spoken reply rejects the entire waveform because thought-to-audio alignment is unverified. Completed reasoning in transcription-only output is removed; PCM16 WAV output is mono 24 kHz and limited to sixty seconds for the complete call turn. An ordinary speech request's non-decoding audio-stream callback stops generation before a 751st 80-ms speech frame (the private codec probe enforces the same bound in its own callback), and returned frame counts are checked again before full codec decoding. Sentence TTS receives only the remaining duration for each segment, so an oversized later segment fails before decoding or publishing it. These limits bound waveform/codec memory and the live-call transport; they are not a 32K speech-output capability claim. Supporting a longer audio output requires a separately designed non-call contract and measured grants, decoding and transport rather than pretending the model context proves fit. The call retains at most twelve recent whole messages without slicing individual messages or applying the native 4,000/8,000-character output ceilings. The worker checks the complete retained prompt against model context and reports overflow rather than silently cropping it. The retired Whisper → language → Qwen call pipeline is removed; standalone inference limits are unchanged. Actual byte/message limits still apply independently of model context, and the advertised streaming modes remain unexposed. The provisional grant is 8 GiB RAM plus 16 GiB VRAM, including model and codec. The short-utterance trial below measured one request; maximum-context fit and coexistence with other models remain unverified.


## Known-script sentence TTS candidate

`HIKARI_LFM3_AUDIO_SENTENCE_TTS=true` on the API selects an additional default-off candidate. It requires single-turn mode and cannot run with the private codec probe. A strict `sentence_tts` task boolean is valid only on complete turns. The same granted disposable child performs recognition, generates a complete text-only answer using the saved call instructions and history, then synthesizes bounded known scripts. Every generation starts with a fresh cache; no model survives the child or its grant.

The text answer uses the ordinary assistant generation prompt with `generation_mode="text"` and speech-mode suppression. It must finish with a valid non-interleaved terminal token. Complete reasoning blocks are removed before any TTS; unfinished reasoning and tool/control output fail. This validation checks completion and privacy, not factual correctness, arithmetic or answer quality.

The worker and controller share the existing sentence/240-character segmentation rule, including its long-word split. Each script uses the pinned card's pure TTS conversation: the American-English female TTS system instruction, the script as the user message and an assistant audio-start prefill. Persona and conversation history remain on the answer stage and are not added to TTS prompts. New speech-mode switches, mixed-mode markers, reasoning and unexpected visible text in TTS output fail before decoding. A valid non-interleaved text-end or turn EOS, nonempty finite 24-kHz PCM and the existing per-output bound are required. All segments together may contain at most sixty seconds; exceeding this produces an error rather than truncation.

Typed `audio_answer` and `audio_segment` events pass only through an opted-in complete-turn job. The controller derives the expected scripts from the public answer, checks contiguous indices and exact scripts, and publishes a completed segment through the existing audio/playback protocol. The complete answer text is sent once with the first valid audio segment, preserving its original formatting; no answer text is displayed before that segment completes synthesis. The final successful job receipt must match the answer, script list, count and all delivered segments before `done` is sent.

Interruption cancels and joins the same child and retains only fully played, acknowledged scripts. A later synthesis failure can leave an already-heard completed prefix, as with native segmented calls. It does not undo heard audio and does not establish that a TTS waveform pronounced its script perfectly. No early codec frames are played, and no text-to-audio word alignment is inferred.

Numeric timings include public-answer generation, first completed TTS segment, total TTS operation time and segment/sample counts, alongside existing setup and ASR timing. `first_tts_segment_seconds` starts after ASR at the reply phase and includes public-answer generation plus the first synthesis; it excludes cold setup and is not total utterance-to-playback latency. More generation calls can make this candidate slower. Real short/multi-sentence playback, interruption, memory and semantic quality require manual validation before any default or readiness change.

## Private streaming codec probe

With single-turn mode enabled, `HIKARI_LFM3_AUDIO_STREAM_PROBE=true` on the API opts into a private decode experiment. It defaults to false. The task carries a strict `codec_streaming_probe` boolean, enabled only for complete turns, and ASR is unchanged. The reply uses the pinned model's `audio_streamer` and its bundled `LiquidCodecStreamingCache`. Non-EOS frames are decoded privately in groups of twelve with one cache and `finalize=False`; pending frames and one empty final flush are processed only after actual turn EOS, complete public-text checks and valid offline audio. The streamer's `end()` merely records completion because the model also calls it after errors and output-cap exhaustion.

The probe holds at most sixty seconds of finite CPU audio and a bounded pending-frame group. It emits only numeric progress (`audio_chunk_decoded`) after the first privately decoded chunk, enabling a cancellation trial during generation. It emits no early text, audio or playback acknowledgement. All model and codec state remains inside the granted disposable child and cancellation still terminates and joins it.

The returned waveform is always the original offline decode. The probe reports first-frame and first-decoded-chunk times from probe construction immediately before reply generation, after reply input preparation. These exclude ASR, cold imports, model/processor loading and therefore are not utterance-to-playback latency. It also reports decoded chunk/sample counts and the maximum absolute float-sample difference against offline decode. Sample-count mismatch or nonfinite/oversized samples fail the probe. Numerical differences are diagnostic; no unmeasured equality tolerance or latency improvement is asserted. Streaming and offline decoding both run in this mode, so it can be slower. Manual measurements and interruption checks are required before changing its default or exposing any streamed playback.

Hikari's `/v1/audio/calls` accepts the ordinary start message with instructions and always selects LFM3-Audio. Kadan's `/api/agents/{key}/calls` resolves the same saved persona snapshot, then relays that start message. The backend selector is removed from both contracts. A normal Kadan frontend exposes **Call** on the agent's pet card; it needs storage/persona availability, not language-model readiness. The utterance, interruption, playback and disconnect protocol is otherwise unchanged. Calls have no extra agent authority or tool access.

## Configuration and existing data

`compose.yaml` includes `hikari-worker-lfm3-audio` without a profile or enable flag. It reuses the former overlay's `lfm3_audio_model_data` volume key and unchanged pinned model ID/revision, so an existing approved download remains usable in the same Compose project. The checkpoint is not downloaded or licensed by startup. `HIKARI_ENABLE_LFM3_AUDIO` and `VITE_ENABLE_LFM3_AUDIO` no longer control availability. The old overlay filename remains usable for existing operator Compose invocations; it now only configures the three retained default-off developer experiments described above.

The former three-model call residency override is removed. LFM audio already uses normal priority 1 and cold executor grants; standalone worker priorities, physical admission and confirmed release rules are unchanged.

Calls have no persistent session/transcript records to migrate. No database tables, native model files, other model volumes or saved transcription preferences are removed. Older clients must update with the API because the removed backend selector is no longer accepted; this does not alter saved personas or conversations. An operator chooses when to rebuild/reload their own services; this source change does not deploy them.

## Owner checks

On the updated local stack, open Kadan's Messages view and the agent's pet card, then choose **Call**. Confirm microphone setup, reply playback, mute, end/disconnect and speaking interruption. Compare with Hikari's independent `/stt` and `/tts` playgrounds and Kadan composer dictation. Record observed results rather than treating readiness or unit tests as inference, audible-quality, full-duplex or maximum-context fit evidence.

The owner reported that the prior LFM audio path worked and accepted merging it. This default-path change has unit/static validation only; it adds no new microphone, listening, playback or hardware acceptance results.

## Historical generated-fixture evidence

The following records come from earlier branch trials. They were not repeated during the master refresh and do not establish current deployment state or human acceptance. No inference, microphone, listening, playback or live-call acceptance checks were performed for this refresh.

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

### Completed-script TTS manual comparison

The separate sentence-TTS trial delivered three complete mono 24-kHz PCM16 WAVs
for a generated utterance requesting a three-sentence arithmetic explanation.
The normal answer prompt produced the correct text answer, 42. First audio
arrived at 17.76 seconds and the final receipt at 22.83 seconds; the text-answer
stage took 8.83 seconds and TTS took 6.98 seconds in total. This was slower than
the same-utterance interleaved sample's 15.11-second buffered reply, which
incorrectly answered 43. These individual samples establish neither general
accuracy nor a latency improvement.

An official direct-response prefill was also tried. It failed the unfinished
reasoning check at 8.94 seconds, before any answer or audio was delivered. The
normal working prompt was restored; the privacy parser was not relaxed.

Interruption after the first completed TTS segment acknowledged cleanup in
0.207 seconds with no late events in the bounded check. No played acknowledgement
was sent for unheard audio. A subsequent ASR roundtrip of the three generated
segments recognized the final result as 32 rather than the intended 42; this
could be synthesis or recognition error and requires listening to resolve.
A short mixed-mode generated reply separately roundtripped as the correct 42.
No microphone was opened, sound played, or audible-playback barge-in verified.
Sentence TTS remains default-off and was disabled after the comparison. The
working normal-prompt implementation is retained for explicit trials of
completed speech segments, not selected as a latency improvement.
