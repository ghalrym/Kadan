# Hikari user expectations

Hikari supplies chat, decisions, images, speech and transcription for Kadan and direct clients. It has no pages of its own: people manage its models in Kadan Server's [AI models](../kadan-server/ai-models.md) section and see its recent work in [Activity](../kadan-server/activity.md). Its abilities are reached by Kadan and by other apps through its API.

Downloading, loading and successful generation are separate states. Requests can wait for hardware or fail when their requirements cannot be met. Each feature describes what its clients can expect while a request waits, fails or is cancelled.

| Feature | What to expect |
| --- | --- |
| [Inference operations](inference.md) | Automatic model loading, memory sharing, priorities and conversation prefix reuse. |
| [Model settings and downloads](model-settings.md) | Fixed model assignments, licenses, and downloads managed from Kadan Server's AI models. |
| [Subscription language providers](subscriptions.md) | Bring a Claude or ChatGPT subscription and switch chat or images between this computer and the subscription. |
| [Model loading and context](model-lifecycle.md) | Check readiness, manage context limits and explicitly load or unload language models. |
| [Decisions](decisions.md) | Evaluate choices, scores and criteria in a single pass using the language model. |
| [Image generation](image-generation.md) | Generate artwork from descriptions and inspect progress and results. |
| [Image editing](image-editing.md) | Edit source images or use reference conditioning with defined preservation limits. |
| [Text to speech](text-to-speech.md) | Generate completed audio using named speakers and delivery instructions. |
| [Transcription](transcription.md) | Transcribe supported recordings directly with Qwen3-ASR 1.7B. |
| [Remote workers](remote-workers.md) | Run workloads on other computers and understand disconnection behavior. |

See [Install and use](../install-and-use.md), [Security and operator responsibility](../security.md) and [Performance tracing](../tracing.md) for shared expectations. [Kadan](../kadan/overview.md) provides agent conversations and actions; [Tsuchi](../tsuchi/overview.md) stores durable content.
