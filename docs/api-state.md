# Settings compatibility and chat history

`GET /v1/settings` reports the actual persisted catalog selection using model IDs
`small`, `medium`, `large`; no selection is `null`. Unsupported modality defaults
are no longer advertised. `PUT /v1/settings` delegates LLM selection to the same
store as `/v1/models/selection`, requiring a completed download and respecting
runtime leases. Supplying transcription formatting returns 503 before mutation:
no transcription provider is configured. The Settings page already uses the
preferred `/v1/models` routes.

`GET /v1/chat/messages` returns an empty list because chat is stateless on the
server. The chat page owns its conversation and sends history in each completion
request. There is no persisted server chat history or canned demonstration chat.
