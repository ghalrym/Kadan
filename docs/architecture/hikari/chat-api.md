# Chat API

`POST /v1/chat/completions` accepts nonempty text messages with `system`, `user`
or `assistant` roles. Each message uses either Hikari's `text` field or the
OpenAI `content` string. JSON responses include `choices` and Hikari's `message`;
SSE responses use content deltas, a `stop` or `length` terminal chunk and `[DONE]`.

Native model requests use a text-only subset, not full OpenAI API compatibility. Tool calls,
`developer`/`tool` roles, content-part arrays, reasoning controls, sampling controls
and streaming usage options are unsupported. Unknown request and message fields
are rejected instead of silently dropping their meaning. Before streaming starts,
errors use FastAPI's HTTP status and `detail` body; stream failures use an `error`
event followed by `[DONE]`, without a successful terminal choice.

Explicit LFM3 requests additionally accept the structured tool history described in [the LFM worker contract](worker-lfm3.md).


## Playground state and retry

The browser sends its full current conversation and receives SSE text. It retains messages and drafts only in page memory. This keeps direct inference separate from Kadan's durable run ownership. See the user expectations for [the chat playground](../../user-expectations/hikari/chat-playground.md).

Hikari Chat explicitly sends `model: lfm3-1.7b-bf16` for every turn, retry and new conversation. That uses the separate LFM worker and its user-approved download. The public endpoint's omitted-model behavior remains the native selected-tier path; this browser default does not change that API contract or voice-call model selection.

A malformed or incomplete stream is a failed turn. Cancel aborts the browser request, discards partial assistant text and retains the user turn for retry. Retry performs another inference rather than restoring an earlier response. Native cleanup can outlast browser cancellation. Model/context selection and memory admission remain in [Model lifecycle](model-lifecycle.md) and [Inference architecture](inference.md).
