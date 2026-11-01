# Chat API

`POST /v1/chat/completions` accepts nonempty text messages with `system`, `user`
or `assistant` roles. Each message uses either Hikari's `text` field or the
OpenAI `content` string. JSON responses include `choices` and Hikari's `message`;
SSE responses use content deltas, a `stop` or `length` terminal chunk and `[DONE]`.

The endpoint uses fixed `lfm3-1.7b-bf16`, whether `model` is supplied or omitted, and rejects other model IDs. When a [subscription provider](subscriptions.md) serves language, the same request is answered by that subscription instead. Structured tool history follows [the LFM worker contract](worker-lfm3.md). Unknown fields, unsupported roles and malformed tool payloads fail validation. Before streaming starts, errors use FastAPI's HTTP status and `detail` body; stream failures use an `error` event followed by `[DONE]`, without a successful terminal choice.

## Playground state and retry

The browser sends its full current conversation and receives SSE text. It retains messages and drafts only in page memory. This keeps direct inference separate from Kadan's durable run ownership. See the user expectations for [the chat playground](../../user-expectations/hikari/chat-playground.md).

Hikari Chat explicitly sends `model: lfm3-1.7b-bf16` for every turn, retry and new conversation. That uses the separate LFM worker and its user-approved download. The same fixed checkpoint serves direct API requests and voice-call language turns.

A malformed or incomplete stream is a failed turn. Cancel aborts the browser request, discards partial assistant text and retains the user turn for retry. Retry performs another inference rather than restoring an earlier response. Native cleanup can outlast browser cancellation. Model/context selection and memory admission remain in [Model lifecycle](model-lifecycle.md) and [Inference architecture](inference.md).
