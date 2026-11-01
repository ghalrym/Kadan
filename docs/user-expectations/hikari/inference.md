# Inference operations

Users expect Hikari to run chat, decisions, speech synthesis, transcription and formatting, image generation, and video generation on their available hardware. Users can request several inference types at once. Hikari manages model loading, memory sharing and waiting work automatically; users should not need to unload one model manually before requesting another type.

## Language models and hardware

### Default LFM3 conversation model

The standard deployment uses a separate released BF16 LFM3-1.7B worker for Hikari Chat and Kadan text conversations. Unavailable workers and incomplete user-approved downloads produce an error, with no substitution. Every other capability also uses its fixed assignment, independent of the computer serving it. The retained native MoE engine is not a selectable conversation model.

This worker loads the model for each request and returns the completed reply through the existing chat interface. Its 262,144-token context is shared by input and output. Generation includes private reasoning that is removed before a reply is returned. By default, generation can use all space remaining after the complete prompt; clients can set smaller generation and reasoning budgets. Reaching the generation budget returns an incomplete response. Output that ends before reasoning closes fails without exposing that reasoning. Prompts leaving no reply space fail without truncating history. Long replies can take much longer and can still fail if memory is insufficient. It does not retain a conversation prefix or support manual preloading. A worker reconnects after losing the service connection, but interrupted work is not automatically replayed. Model quality, latency and physical fit require owner validation on the target computer; a connected worker or downloaded checkpoint does not prove successful inference.

## Multiple inference types at once

Requests for different inference types can execute concurrently when the hardware has enough memory. Accepting several requests does not promise that they all start immediately. When their combined requirements exceed available memory, some requests wait while others run.

A worker container's memory limit constrains its own work. It must not reduce the physical memory available to other workers on the same computer; their combined allocations must still fit the computer's available memory.

When requests compete for memory, Hikari uses this priority order, highest first:

1. Chat and decisions.
2. Speech synthesis.
3. Transcription and transcription formatting.
4. Image generation.
5. Video generation.

A higher-priority request does not interrupt inference already running. For example, chat and image generation can overlap when both fit. If a video is already running and occupies the memory chat needs, chat waits rather than cancelling the video.

Hikari tries to keep room for chat while other work runs, but chat has no unconditional immediate-start guarantee. Sustained higher-priority demand can delay lower-priority work.

## Loading, waiting and completion

Models load on demand. A first request or returning to a previously unloaded inference type can incur model-loading time. Switching workloads under memory pressure can also require a model to reload. Hikari handles this automatically.

Models retained between requests unload after one hour without work, measured from the last request's completion. LFM conversation and screenshot executors instead release their models after each request. Operators can configure the retained-model idle period; see [inference architecture](../../architecture/hikari/inference.md). Idle expiry never interrupts active or queued requests. Explicit unloading and requests that need memory can release a model sooner. A later request reloads it; downloaded checkpoints and conversation history are unchanged.

A queued request has been accepted, not completed. Hikari must not silently truncate chat history or reduce requested media settings to force a fit. Invalid inputs, unavailable checkpoints and execution failures must produce explicit errors rather than fabricated results. Cancellation must finish cleanup before the released resources are reused. If a worker cannot confirm cleanup, its affected capacity stays unavailable and may require a worker restart. An uncertain operation is not automatically replayed.

Concurrent throughput, model quality and physical memory fit require manual validation on the target machine. Hikari does not guarantee a fixed response time or that every combination of inference requests fits.

## Conversation prefix reuse

LFM conversations retain no reusable prefix between requests. Language requests served by the retained native worker can reuse its last completed prefix when the earlier prompt exactly starts a longer one. Direct clients can disable reuse; requests without a conversation identity do not retain a prefix. Edited, shorter or identical prompts, another conversation, cancellation and incompatible model or context settings require a fresh prefill. A different worker may also miss the cache.

When an idle model unloads or yields VRAM, Hikari can keep its reusable KV and recurrent state in RAM. A compatible later request restores that state after loading the model. Each worker retains at most one prefix; it can discard it sooner when RAM is needed for work. No reusable state is written to disk, and restarting the worker loses it.

The prefix cache expires after one hour without reuse by default. Operators can configure this separately from model idle expiry: unloading a model does not restart the cache timer, and discarding its cache does not unload the model. Expiry waits for active and queued work. Conversation history and downloaded checkpoints are unaffected. Responses report actual reused and stored tokens and whether a prefix was restored from RAM; reuse does not guarantee a measured latency improvement.

## Vision subscriptions on this branch

The Hikari backend can answer screenshot questions through the independently selected Claude or ChatGPT subscription, without a local Vision checkpoint or memory allocation. Local behavior is unchanged. Provider failures are explicit and never trigger fallback; see [subscription expectations](subscriptions.md#vision-subscription-backend-on-this-branch). This branch behavior has not been verified on master.
