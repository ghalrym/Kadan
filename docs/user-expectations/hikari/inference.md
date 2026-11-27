# Inference operations

Users expect Hikari to run chat, decisions, speech synthesis, transcription and formatting, and image generation on their available hardware. Users can request several inference types at once. Hikari manages model loading, memory sharing and waiting work automatically; users should not need to unload one model manually before requesting another type.

## Language models and hardware

### Inkling 2 Mini language and vision model

The standard deployment uses Inkling 2 Mini running natively on Hikari's Strata MoE engine for Hikari Chat and Kadan agent conversations. Unavailable workers and incomplete user-approved downloads produce an error, with no substitution. Every other capability also uses its fixed assignment, independent of the computer serving it.

Generation includes private reasoning that is separated from the visible reply. By default, generation can use the space remaining after the complete prompt; clients can set smaller generation and reasoning budgets. Reaching the generation budget returns an incomplete response. Prompts leaving no reply space fail without truncating history. Long replies can take longer and can still fail if memory is insufficient. A worker reconnects after losing the service connection, but interrupted work is not automatically replayed. Model quality, latency and physical fit require owner validation on the target computer; a connected worker or downloaded checkpoint does not prove successful inference.

## Multiple inference types at once

Requests for different inference types can execute concurrently when the hardware has enough memory. Accepting several requests does not promise that they all start immediately. When their combined requirements exceed available memory, some requests wait while others run.

A worker container's memory limit constrains its own work. It must not reduce the physical memory available to other workers on the same computer; their combined allocations must still fit the computer's available memory.

When requests compete for memory, Hikari uses this priority order, highest first:

1. Chat and decisions.
2. Speech synthesis.
3. Transcription and transcription formatting.
4. Image generation.

A higher-priority request does not interrupt inference already running. For example, chat and image generation can overlap when both fit. If an image generation is already running and occupies the memory chat needs, chat waits rather than cancelling the image generation. Within queued language requests, interactive chat replies take priority over background computer work between turns, while ensuring waiting background work still makes progress.

Hikari tries to keep room for chat while other work runs, but chat has no unconditional immediate-start guarantee. Sustained higher-priority demand can delay lower-priority work.

## Loading, waiting and completion

Models load on demand. A first request or returning to a previously unloaded inference type can incur model-loading time. Switching workloads under memory pressure can also require a model to reload. Hikari handles this automatically.

Models retained between requests unload after one hour without work, measured from the last request's completion. Operators can configure the retained-model idle period; see [inference architecture](../../architecture/hikari/inference.md). Idle expiry never interrupts active or queued requests. Explicit unloading and requests that need memory can release a model sooner. A later request reloads it; downloaded checkpoints and conversation history are unchanged.

A queued request has been accepted, not completed. Hikari must not silently truncate chat history or reduce requested media settings to force a fit. Invalid inputs, unavailable checkpoints and execution failures must produce explicit errors rather than fabricated results. Cancellation must finish cleanup before the released resources are reused. If a worker cannot confirm cleanup, its affected capacity stays unavailable and may require a worker restart. An uncertain operation is not automatically replayed.

Concurrent throughput, model quality and physical memory fit require manual validation on the target machine. Hikari does not guarantee a fixed response time or that every combination of inference requests fits.

## Conversation prefix reuse

Language requests served by Inkling 2 Mini can reuse an earlier matching prompt prefix (including matching image attachments when identical, or stopping before any differing image). Direct clients can disable reuse; requests without a conversation identity do not retain conversation-specific state, though identical shared system instructions can still be reused. Edited, shorter or identical prompts, cancellation and incompatible model or context settings require a fresh prefill. A different worker may also miss the cache.

When switching between conversations or when an idle model unloads or yields memory, Hikari can keep compacted prefix checkpoints in memory within its fixed memory budget so recent conversations—such as chat and computer work—can resume without starting from scratch. Older checkpoints are evicted as memory fills, and checkpoints can be discarded sooner when memory is needed for active work. No reusable state is written to disk, and restarting loses it.

The prefix cache expires after one hour without reuse by default. Operators can configure this separately from model idle expiry: unloading a model does not restart the cache timer, and discarding its cache does not unload the model. Expiry waits for active and queued work. Conversation history and downloaded checkpoints are unaffected. Responses report actual reused and stored tokens and whether a prefix was restored from RAM; reuse does not guarantee a measured latency improvement.
