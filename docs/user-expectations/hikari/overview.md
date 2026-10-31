# Hikari user expectations

Hikari supplies chat, decisions, images, video, speech and transcription for Kadan and direct clients. Its management interface lets users download its fixed models, try inference and inspect recent requests.

Downloading, loading and successful generation are separate states. Requests can wait for hardware or fail when their requirements cannot be met. Playground results generally belong to the open page; each feature describes what survives navigation or a service restart.

| Feature | What to expect |
| --- | --- |
| [Inference operations](inference.md) | Automatic model loading, memory sharing, priorities and conversation prefix reuse. |
| [Model settings and downloads](model-settings.md) | Review fixed assignments and licenses and track or resume downloads. |
| [Model loading and context](model-lifecycle.md) | Check readiness, manage context limits and explicitly load or unload language models. |
| [Chat playground](chat-playground.md) | Try streaming text conversations with cancellation and retry controls. |
| [Decisions](decisions.md) | Evaluate choices, scores and criteria using specialist model judgments. |
| [Image generation](image-generation.md) | Generate artwork from descriptions and inspect progress and results. |
| [Image editing](image-editing.md) | Edit source images or use reference conditioning with defined preservation limits. |
| [Video generation](video-generation.md) | Queue clips with audio, cancel jobs and download completed media. |
| [Text to speech](text-to-speech.md) | Generate completed audio using named speakers and delivery instructions. |
| [Transcription](transcription.md) | Transcribe supported recordings with optional formatting and raw-text fallback. |
| [Live voice calls](voice-calls.md) | Build direct spoken conversations with client-managed turns, playback and acknowledgements. |
| [Request monitoring](request-monitoring.md) | Inspect bounded recent request summaries, timings and resource observations. |
| [API access](api-access.md) | Discover the running service's contract and submit direct inference requests. |
| [Remote workers](remote-workers.md) | Run workloads on other computers and understand disconnection behavior. |

See [Install and use](../install-and-use.md), [Security and operator responsibility](../security.md) and [Performance tracing](../tracing.md) for shared expectations. [Kadan](../kadan/overview.md) provides agent conversations and actions; [Tsuchi](../tsuchi/overview.md) stores durable content.
