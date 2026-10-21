# Inference operations

Users expect Hikari to run chat, decisions, speech synthesis, transcription and formatting, image generation, and video generation on their available hardware. Users can request several inference types at once. Hikari manages model loading, memory sharing and waiting work automatically; users should not need to unload one model manually before requesting another type.

## Language models and hardware

Hikari can run the selected language checkpoint even when the complete model is larger than GPU memory, provided the computer has enough memory for the required model weights and the request. It must not silently substitute a smaller or dense language model to make a request fit.

Using system memory or disk can make responses slower. Available memory, CPU speed, storage speed and GPU transfer speed all affect performance. A hardware tier is a starting configuration, not a guarantee of response speed or of fitting every context length.

## Multiple inference types at once

Requests for different inference types can execute concurrently when the hardware has enough memory. Accepting several requests does not promise that they all start immediately. When their combined requirements exceed available memory, some requests wait while others run.

When requests compete for memory, Hikari uses this priority order, highest first:

1. Chat and decisions.
2. Speech synthesis.
3. Transcription and transcription formatting.
4. Image generation.
5. Video generation.

A higher-priority request does not interrupt inference already running. For example, chat and image generation can overlap when both fit. If a video is already running and occupies the memory chat needs, chat waits rather than cancelling the video.

Hikari tries to keep room for chat while other work runs, but chat has no unconditional immediate-start guarantee. Sustained higher-priority demand can delay lower-priority work.

## Loading, waiting and completion

Models load on demand. A first request, a change of hardware tier, or returning to a previously unloaded inference type can incur model-loading time. Switching workloads under memory pressure can also require a model to reload. Hikari handles this automatically.

A queued request has been accepted, not completed. Hikari must not silently truncate chat history or reduce requested media settings to force a fit. Invalid inputs, unavailable checkpoints and execution failures must produce explicit errors rather than fabricated results. Cancellation must finish cleanup before the released resources are reused.

Concurrent throughput, model quality and physical memory fit require manual validation on the target machine. Hikari does not guarantee a fixed response time or that every combination of inference requests fits.
