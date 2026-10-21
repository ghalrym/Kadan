# Chat playground

Hikari's Chat page tests text inference directly. It starts empty and keeps its conversation and draft only in the open page's memory. Reloading or leaving the page loses that history. It is separate from Kadan's durable agent conversations and does not execute agent tools or save messages in Tsuchi.

Enter sends a nonblank message; Shift+Enter inserts a newline. One response runs at a time. The page sends its full current conversation to Hikari and shows streamed assistant text. Model selection comes from Hikari's saved configuration rather than a model picker in this composer. A missing tier or failed checkpoint acquisition produces an error; a downloaded model can still require loading time.

Conversation capacity depends on the selected model's configured token context. The displayed character count is not a token budget. Requests that exceed capacity must fail explicitly rather than silently dropping old messages. A response can also reach the model's output limit; receiving text is not a guarantee that every requested detail was completed.

Cancel stops accepting the response in this browser and removes its partial assistant text, retaining the submitted conversation for Retry last message. Backend cleanup can finish later. Leaving the page also aborts the browser request. This differs from leaving Kadan chat, where saved work continues.

A failed or cancelled turn must be retried or cleared with New chat before another message can be sent. Retry last message reuses the same user turn in the displayed conversation and submits inference again; it is not a guarantee of identical output. New chat clears this page's messages and draft. An incomplete or malformed stream is an error, not a completed assistant message.

API clients have the text-only compatibility subset described in [Chat API scope](../../../services/hikari/README.md#chat-api-scope). Conversation prefix reuse can reduce repeated prefill under the conditions in [Inference operations](inference.md#conversation-prefix-reuse); it does not make this playground a durable conversation store.
