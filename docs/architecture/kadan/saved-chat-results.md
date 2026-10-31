# Saved chat results

See [Saving replies to Workspace](../../user-expectations/kadan/saving-chat-results.md) for user-visible behavior.

Kadan exposes a title-only action for turning a stored assistant reply into a Tsuchi Space document. The storage transaction reads the active same-agent conversation and copies the reply's Markdown, source links and attachment references. It adds source conversation/message tags; it does not copy attachment bytes or execute the reply's instructions.

Tsuchi derives the document UUID from the conversation UUID and message UUID and creates it under its existing write transaction. Repeating a request after a lost receipt returns the same document without replacing its current title, content or revision history. A trashed result produces a conflict requiring restoration in Tsuchi. No idempotency table or secondary artifact store is introduced.

The document is an independent initial snapshot. Later conversation changes do not update it, and later document edits do not change the conversation. Saving does not enqueue a run or provide automatic recall. General editing and revision restoration stay in Tsuchi. This API must be deployed with the matching Tsuchi route; an older storage service returns an ordinary visible request failure.
