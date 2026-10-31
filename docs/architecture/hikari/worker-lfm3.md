# LFM3 conversation worker

Andrew's later migration direction makes released BF16 LFM3 the conversation default in the normal Compose deployment. Separate Python worker kinds use the existing Hikari queue, worker authentication, per-computer memory ledger and model download endpoints. They do not enter the native C++ executors. Existing language, speech, image and video workers remain available for their remaining consumers; audio migration is separate.

Each inference runs in a spawned Python executor after its worker receives a memory grant. Cancellation terminates and joins that executor before reporting cleanup. It loads on every request and retains no CUDA model or conversation-prefix state between requests. Fixed conservative resource reservations and bounded input/context sizes make admission reviewable, but memory fit and speed still depend on the target hardware and request.

Normal Compose starts both text and vision workers without profiles and defaults their Hikari API/worker enable flags to true. Kadan always uses the fixed language and vision assignments without separate selection flags. Existing explicit environment overrides still apply. Hikari Chat always sends the released text model ID, including retries and new conversations; it never falls back to the native model. Public API requests omitting a model also use fixed LFM3; other language IDs are rejected. Separately launched services still require their matching flags. Audio remains opt-in. Calls use fixed LFM3, Whisper medium and Qwen CustomVoice; the native engines remain retained.

The text checkpoint is `LiquidAI/LFM3-1.7B` at `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, with BF16 weights and checkpoint custom code. The artifact manifest pins sizes and content hashes, including code, tokenizer, config and license. Loading uses only the verified local snapshot; it cannot resolve a mutable upstream branch or download a replacement during inference. The user must review and acknowledge the LFM Open License v1.0 through the existing download flow. Enabling a container does not acknowledge it.

Text requests name `lfm3-1.7b-bf16` explicitly. The API and worker require `HIKARI_ENABLE_LFM3_TEXT=true`. Kadan always uses this checkpoint for agent conversation and readiness, preserving existing session and tool orchestration. Only this model accepts structured `tools`, assistant `tool_calls` with mapping arguments, and `tool` results with their call identities. Kadan passes these to the [pinned checkpoint chat template](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd/chat_template.jinja) instead of injecting the native Qwen tool preamble or disguising tool results as user turns. Its output XML syntax matches the existing parser; Kadan still validates and records a command before execution. The template is applied with thinking disabled, and returned reasoning is removed before tool parsing. Truncated generations remain incomplete commands. Native requests retain their existing text-only contract.

The text trial's total context is 262,144 tokens, including input and output. Generation can use all positions remaining after the complete templated prompt, with no separate 1,024-token output cap. A prompt filling the context is rejected instead of cropped. An actual EOS token completes the turn, including at the context boundary; exhausting the remaining positions without EOS is incomplete. Public manual preload and saved-context controls are removed; the retained native context code does not configure this per-request executor. This removes an artificial reply ceiling, not a guarantee of useful or fast maximum-length generation: memory admission, cancellation and Kadan's command-size validation remain in force.

Generation uses Transformers 5.21's supported `prefill_chunk_size=512` with a dynamic hybrid cache, caching enabled and compilation disabled. The checkpoint delegates its prefill to that implementation, which carries the convolution, sliding and full-attention cache between chunks, slices the cumulative attention mask and absolute positions, and uses the final chunk's logits for the first generated token. The complete prompt is passed once to `generate`; the application does not slice away history or prefill tokens twice. This bounds each SDPA mask's query axis rather than allocating one full-window square mask. Efficient CUDA attention kernels are required; math-attention fallback is disabled because its score allocation can exceed the grant. Cache growth and physical fit still require target-machine validation and can fail explicitly.

See [inference expectations](../../user-expectations/hikari/inference.md) for the observable trial limits. Operation and rollback commands are in the [shared worker setup](../../../services/hikari/experimental_worker/README.md).

## Manual trial evidence

An owner-authorized BF16 trial on one RTX 3090 completed a real Kadan reply and
read-only screenshot delegation. Short coordinator requests generated 20–48
tokens in 1.61–2.21 seconds including prefill, with cold executor times of
6.41–7.00 seconds. These counts include model control tokens; they are not
decode-only throughput or a general tool-quality evaluation.

A separate capacity probe submitted exactly 261,120 prompt tokens with the
1,024-token reply reservation. It returned both boundary markers, generated
90 tokens and finished in 100.43 seconds including cold loading. Peak CUDA
allocation was 8.61 GiB, reservation 18.65 GiB, and sampled device use
19,352 MiB, within the 20 GiB grant. This establishes one full-window fit and
marker retrieval, not general long-context quality or simultaneous full-window
capacity. Oversized input was separately rejected without truncation.

Executor measurements report elapsed time, peak RSS and CUDA allocation and
reservation. Generation timing excludes model loading but includes prefill.
They do not include prompt or reply content. Workers reconnect with capped
backoff after transport loss, refresh resource identity and do not replay
unconfirmed work. Workers restart unless explicitly stopped.

Executor teardown runs in a separate task that is shielded from consumer
cancellation. The consumer keeps awaiting it after further cancellation requests,
then clears its child and budget only after physical exit and join. Cancellation
propagates afterward, so a disconnect cannot interrupt an existing termination
wait and let the reconnect loop admit work beside a surviving executor.
