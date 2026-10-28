# Conversation worker (`lfm3_llm`)

The fixed conversation checkpoint is BF16 LFM3-1.7B, served by the Python LFM worker. The former native Qwen mixture-of-experts language worker and hardware-tier selection are retired. [LFM worker architecture](worker-lfm3.md) owns the pinned checkpoint, tool-template, context and generation details.

The API routes omitted model fields to that same checkpoint and rejects unsupported alternate language models. Conversation and transcript formatting share its queue and admission priority. Agent tools remain owned by Kadan; the inference worker formats structured tool history and generates text but does not execute agent actions.

Each request loads a disposable executor after a RAM and single-GPU grant. Completion and cancellation wait for process exit before returning the allocation. The executor retains no language model or prefix cache across requests, and manual preloading is unavailable. Saved per-model context limits affect subsequent requests and never silently trim the prompt.

The declared resource envelope is conservative admission metadata, not a measured guarantee that the maximum window or concurrent workloads fit. Native image and video engines retain their independent memory and residency behavior; see [shared scheduling](inference.md).
