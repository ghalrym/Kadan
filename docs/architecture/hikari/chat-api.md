# Chat API

`POST /v1/chat/completions` accepts nonempty messages with `system`, `user`
or `assistant` roles. Each message uses either Hikari's `text` field, an
OpenAI `content` string, or OpenAI-style multimodal `content` parts containing
text and `image_url` payloads. JSON responses include `choices` and Hikari's `message`;
SSE responses use content deltas, a `stop` or `length` terminal chunk and `[DONE]`.

The endpoint uses the configured local Inkling 2 Mini checkpoint (`inkling-2-mini-iq4-nl` by default, or `inkling-2-mini-q8-0` via `HIKARI_LLM_QUANT`), whether `model` is supplied or omitted, and rejects unsupported model IDs. When a [subscription provider](subscriptions.md) serves language, the same request (including multimodal screenshots) is answered by that subscription instead. Structured tool definitions and tool history follow [the language worker contract](worker-llm.md). Unknown fields, unsupported roles and malformed tool or image payloads fail validation. Before streaming starts, errors use FastAPI's HTTP status and `detail` body; stream failures use an `error` event followed by `[DONE]`, without a successful terminal choice.

## Playground state and retry

The browser sends its full current conversation and receives SSE text. It retains messages and drafts only in page memory. This keeps direct inference separate from Kadan's durable run ownership. See the user expectations for [the chat playground](../../user-expectations/hikari/chat-playground.md).

Hikari Chat sends the active Inkling 2 Mini checkpoint (or lets Hikari route to the configured language subscription) for every turn, retry and new conversation. That uses the native `hikari-worker-llm` Strata worker and its user-approved download when local.

A malformed or incomplete stream is a failed turn. Cancel aborts the browser request, discards partial assistant text and retains the user turn for retry. Retry performs another inference rather than restoring an earlier response. Native cleanup can outlast browser cancellation. Model/context selection and memory admission remain in [Model lifecycle](model-lifecycle.md) and [Inference architecture](inference.md).
