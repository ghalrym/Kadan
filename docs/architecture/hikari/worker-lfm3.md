# LFM3 workers

The fixed BF16 stack uses separate conversation, screenshot and audio worker kinds. They use the existing Hikari queue, worker authentication, per-computer memory ledger and model download endpoints. LFM inference does not use the retired native C++ language executors. Qwen Image Turbo generation/editing and H3 media components remain available.

Each inference runs in a spawned Python executor after its worker receives a memory grant. Cancellation terminates and joins that executor before reporting cleanup. The initial experiment loads on every request and retains no CUDA model or conversation-prefix state between requests. Fixed conservative resource reservations and bounded input/context sizes make admission reviewable, but memory fit and speed still depend on the target hardware and request.

The text checkpoint is `LiquidAI/LFM3-1.7B` at `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, with BF16 weights and checkpoint custom code. The artifact manifest pins sizes and content hashes, including code, tokenizer, config and license. Loading uses only the verified local snapshot; it cannot resolve a mutable upstream branch or download a replacement during inference. The user must review and acknowledge the LFM Open License v1.0 through the existing download flow. Enabling a container does not acknowledge it.

Text requests name `lfm3-1.7b-bf16` explicitly. Omitted language-model fields select that same fixed checkpoint. Kadan checks actual checkpoint availability and download state, preserving existing sessions and tool orchestration. The language endpoint accepts structured `tools`, assistant `tool_calls` with mapping arguments, and `tool` results with their call identities. Kadan passes these to the [pinned checkpoint chat template](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd/chat_template.jinja) instead of injecting the native Qwen tool preamble or disguising tool results as user turns. Its output XML syntax matches the existing parser; Kadan still validates and records a command before execution. The template is applied with thinking disabled, and returned reasoning is removed before tool parsing. Truncated generations remain incomplete commands. Unsupported alternate language models are rejected.

The text trial's total context is 262,144 tokens, including input and output. Generation can use all positions remaining after the complete templated prompt, with no separate 1,024-token output cap. A prompt filling the context is rejected instead of cropped. An actual EOS token completes the turn, including at the context boundary; exhausting the remaining positions without EOS is incomplete. The saved per-model context limit configures subsequent executors; manual preloading is unavailable. This removes an artificial reply ceiling, not a guarantee of useful or fast maximum-length generation: memory admission, cancellation and Kadan's command-size validation remain in force.

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
unconfirmed work. Compose workers restart unless explicitly stopped.
