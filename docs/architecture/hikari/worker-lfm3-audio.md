# LFM3 audio worker

The fixed speech and call worker runs released BF16 [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B), pinned to `9d49701c372d6011d900f6abb40c6b296fb706dd`. Its manifest verifies model, codec, tokenizer, configuration and custom-code files. Hikari retains its explicit license acknowledgement and download flow. Native Whisper file transcription, image and video workers remain separate.

The Python worker uses the checkpoint's pinned custom Transformers implementation, local-only loading, BF16, one CUDA GPU and bundled LiquidCodec. It replaces Qwen speech and call generation in this migration. No API process imports or runs the inference libraries.

`AudioCallTask` routes to `lfm3_audio` through normal memory admission. The default `turn` task uses one disposable child to load once, transcribe the utterance and generate the buffered reply with the same model, processor and codec. Each generation starts with a fresh cache; ASR state is not reused for a differently prompted reply. Cancellation terminates and joins the child before releasing memory. `HIKARI_LFM3_AUDIO_SINGLE_TURN=false` retains a diagnostic two-job path that loads once for recognition and again for reply. Neither path retains an engine or prefix cache between turns.

Standalone `SpeechTask` uses the same owner and checkpoint with the pinned pure-TTS prompt. It accepts the model card's supported locale and female/male controls, returns a completed mono 24-kHz WAV, and rejects incomplete or oversized output. It provides no Qwen named voices or voice cloning. See [speech architecture](worker-tts.md).

The single-turn path emits a typed `audio_transcript` worker event before reply generation. Hikari forwards this through a bounded audio event queue, remembers the user text and delivers the ordinary transcript event immediately. A later reply failure or interruption preserves that recognized utterance. The worker and call controller use the same pure history-retention function when constructing the reply context, so fusing the operations does not change which prior messages are included. Reply text and PCM still wait for the existing EOS and privacy checks, and only a played, acknowledged reply enters assistant history. Cancelling the stream uses the existing cancel-and-join path; no new utterance is acknowledged before cleanup.

The adapter accepts only mono PCM16 16-kHz audio, at most twenty seconds. Its full released 32,768-token audio context is shared by input and generated slots. Each operation permits every remaining slot after the exact tokenized prompt, including audio placeholders, instead of reserving a fixed 1,024-token reply. A prompt occupying the whole context fails because no reply slot remains; no retained input is cropped to fit. The checkpoint exports a 128,000-position text backbone; that does not establish a larger supported multimodal context than the audio model card advertises. Replies require the mixed turn's EOS/im_end plus both text and speech. The text-end marker starts the audio tail and cannot complete a reply; transcription-only output can end at text-end. Raw decoded output is checked before special markers are removed: unfinished reasoning and tool markup are rejected, and any reasoning in a spoken reply rejects the entire waveform because thought-to-audio alignment is unverified. Completed reasoning in transcription-only output is removed; PCM16 WAV output is mono 24 kHz and limited to sixty seconds for the complete call turn. An ordinary speech request's non-decoding audio-stream callback stops generation before a 751st 80-ms speech frame (the private codec probe enforces the same bound in its own callback), and returned frame counts are checked again before full codec decoding. Sentence TTS receives only the remaining duration for each segment, so an oversized later segment fails before decoding or publishing it. These limits bound waveform/codec memory and the live-call transport; they are not a 32K speech-output capability claim. Supporting a longer audio output requires a separately designed non-call contract and measured grants, decoding and transport rather than pretending the model context proves fit. The call retains at most twelve recent whole messages without slicing individual messages or applying the native 4,000/8,000-character output ceilings. The worker checks the complete retained prompt against model context and reports overflow rather than silently cropping it. Actual byte/message limits still apply independently of model context, and the advertised streaming modes remain unexposed. The provisional grant is 8 GiB RAM plus 16 GiB VRAM, including model and codec. The short-utterance trial below measured one request; maximum-context fit and coexistence with other models remain unverified.



## Known-script sentence TTS candidate

`HIKARI_LFM3_AUDIO_SENTENCE_TTS=true` on the API selects an additional default-off candidate. It requires single-turn mode and cannot run with the private codec probe. A strict `sentence_tts` task boolean is valid only on complete turns. The same granted disposable child performs recognition, generates a complete text-only answer using the saved call instructions and history, then synthesizes bounded known scripts. Every generation starts with a fresh cache; no model survives the child or its grant.

The text answer uses the ordinary assistant generation prompt with `generation_mode="text"` and speech-mode suppression. It must finish with a valid non-interleaved terminal token. Complete reasoning blocks are removed before any TTS; unfinished reasoning and tool/control output fail. This validation checks completion and privacy, not factual correctness, arithmetic or answer quality.

The worker and controller share the existing sentence/240-character segmentation rule, including its long-word split. Each script uses the pinned card's pure TTS conversation: the American-English female TTS system instruction, the script as the user message and an assistant audio-start prefill. Persona and conversation history remain on the answer stage and are not added to TTS prompts. New speech-mode switches, mixed-mode markers, reasoning and unexpected visible text in TTS output fail before decoding. A valid non-interleaved text-end or turn EOS, nonempty finite 24-kHz PCM and the existing per-output bound are required. All segments together may contain at most sixty seconds; exceeding this produces an error rather than truncation.

Typed `audio_answer` and `audio_segment` events pass only through an opted-in complete-turn job. The controller derives the expected scripts from the public answer, checks contiguous indices and exact scripts, and publishes a completed segment through the existing audio/playback protocol. The complete answer text is sent once with the first valid audio segment, preserving its original formatting; no answer text is displayed before that segment completes synthesis. The final successful job receipt must match the answer, script list, count and all delivered segments before `done` is sent.

Interruption cancels and joins the same child and retains only fully played, acknowledged scripts. A later synthesis failure can leave an already-heard completed prefix. It does not undo heard audio and does not establish that a TTS waveform pronounced its script perfectly. No early codec frames are played, and no text-to-audio word alignment is inferred.

Numeric timings include public-answer generation, first completed TTS segment, total TTS operation time and segment/sample counts, alongside existing setup and ASR timing. `first_tts_segment_seconds` starts after ASR at the reply phase and includes public-answer generation plus the first synthesis; it excludes cold setup and is not total utterance-to-playback latency. More generation calls can make this candidate slower. Real short/multi-sentence playback, interruption, memory and semantic quality require manual validation before any default or readiness change.

## Private streaming codec probe

With single-turn mode enabled, `HIKARI_LFM3_AUDIO_STREAM_PROBE=true` on the API opts into a private decode experiment. It defaults to false. The task carries a strict `codec_streaming_probe` boolean, enabled only for complete turns, and ASR is unchanged. The reply uses the pinned model's `audio_streamer` and its bundled `LiquidCodecStreamingCache`. Non-EOS frames are decoded privately in groups of twelve with one cache and `finalize=False`; pending frames and one empty final flush are processed only after actual turn EOS, complete public-text checks and valid offline audio. The streamer's `end()` merely records completion because the model also calls it after errors and output-cap exhaustion.

The probe holds at most sixty seconds of finite CPU audio and a bounded pending-frame group. It emits only numeric progress (`audio_chunk_decoded`) after the first privately decoded chunk, enabling a cancellation trial during generation. It emits no early text, audio or playback acknowledgement. All model and codec state remains inside the granted disposable child and cancellation still terminates and joins it.

The returned waveform is always the original offline decode. The probe reports first-frame and first-decoded-chunk times from probe construction immediately before reply generation, after reply input preparation. These exclude ASR, cold imports, model/processor loading and therefore are not utterance-to-playback latency. It also reports decoded chunk/sample counts and the maximum absolute float-sample difference against offline decode. Sample-count mismatch or nonfinite/oversized samples fail the probe. Numerical differences are diagnostic; no unmeasured equality tolerance or latency improvement is asserted. Streaming and offline decoding both run in this mode, so it can be slower. Manual measurements and interruption checks are required before changing its default or exposing any streamed playback.

The Hikari call start defaults to the sole supported `backend: "lfm3"`. Kadan's ordinary **Call** action resolves the saved persona snapshot and uses this model; no extra enable flag or experimental button is required. The existing utterance, interruption, playback and disconnect protocol remains authoritative. A buffered reply is one acknowledged segment, and only acknowledged assistant text enters later history. There is no extra agent authority or tool access.

## Manual validation

Use the [normal worker setup](../../../services/hikari/experimental_worker/README.md) with the existing Compose project and URLs. The audio checkpoint keeps its own persistent volume. Review the linked license and code notice, explicitly confirm its download through Settings and wait for completed status. Starting a worker or running unit tests does not download a model or accept a license.

1. Open **Call** using HTTPS or localhost and browser microphone permission. Speak a short English utterance, inspect transcript and reply, and listen to the complete playback. Record load time, peak RAM/VRAM and perceived speech quality.
2. Interrupt while recognition or generation is pending. Check cleanup completes before the next turn and stale audio is discarded. Interrupt partly played speech, then compare with a fully acknowledged reply to verify retained context.
3. Exercise mute, disconnect and a new call; check history resets and image/video workers remain available. Inspect jobs after uncertain outcomes before retrying.
4. Compare the same utterance with single-turn mode enabled and disabled only when diagnosing latency. Record actual timing and memory; one fewer load is not proof of live usability.
5. Exercise standalone speech locales and both gender controls, listen for correspondence with the requested script, and check cancellation and explicit duration-limit errors.

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
