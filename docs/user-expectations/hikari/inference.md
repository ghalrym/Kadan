# Inference operations

Users expect Hikari to run chat, decisions, speech synthesis, transcription and formatting, image generation, and video generation on their available hardware. Users can request several inference types at once. Hikari manages model loading, memory sharing and waiting work automatically; users should not need to unload one model manually before requesting another type.

## Language models and hardware

Hikari uses the fixed LFM conversation checkpoint and must not silently substitute another model to make work fit. BF16 execution needs compatible hardware and enough per-device memory for the checkpoint, prompt and reply. A supported context window does not guarantee that every request fits.

The conversation model's 262,144-token context is shared by input and output. Replies can use the space remaining after the complete prompt, until the model finishes or the request is cancelled. Prompts leaving no reply space fail without truncating history. Each request currently loads its own model; no conversation prefix or manual preload is retained. Interrupted work is not automatically replayed.

## Multiple inference types at once

Requests for different inference types can execute concurrently when the hardware has enough memory. Accepting several requests does not promise that they all start immediately. When their combined requirements exceed available memory, some requests wait while others run.

When requests compete for memory, Hikari uses this priority order, highest first:

1. LFM conversation, screenshot understanding, speech and transcript formatting, plus decisions.
2. Native transcription.
3. Image generation.
4. Video generation.

A higher-priority request does not interrupt inference already running. For example, chat and image generation can overlap when both fit. If a video is already running and occupies the memory chat needs, chat waits rather than cancelling the video.

Hikari tries to keep room for chat while other work runs, but chat has no unconditional immediate-start guarantee. Sustained higher-priority demand can delay lower-priority work.

## Loading, waiting and completion

Models load on demand. LFM requests currently load a fresh executor for each request; returning to an unloaded specialist engine can also incur model-loading time. Switching workloads under memory pressure can also require a model to reload. Hikari handles this automatically.

Retained specialist engines unload their resident model after one hour without work, measured from the last request's completion. LFM executors release their models after each request. Operators can configure this period; see [inference architecture](../../architecture/hikari/inference.md). Idle expiry never interrupts active or queued requests. Explicit unloading and requests that need memory can release a model sooner. A later request reloads it; downloaded checkpoints and conversation history are unchanged.

A queued request has been accepted, not completed. Hikari must not silently truncate chat history or reduce requested media settings to force a fit. Invalid inputs, unavailable checkpoints and execution failures must produce explicit errors rather than fabricated results. Cancellation must finish cleanup before the released resources are reused.

Concurrent throughput, model quality and physical memory fit require manual validation on the target machine. Hikari does not guarantee a fixed response time or that every combination of inference requests fits.

## Conversation prefix reuse

The current LFM conversation executor releases its model and cache after every request. Conversation history is sent again for each turn; no prefix reuse or retained resident model is promised. Cancellation and completion release resources only after executor cleanup.
