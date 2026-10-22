# Chat notes

Kadan separates storing a user message from requesting agent work. This allows notes to use the same conversation history without requiring inference. See [Saving chat notes](../../user-expectations/kadan/saving-chat-notes.md) for the user-facing behavior.

The note path appends a user message through Tsuchi without enqueuing an agent run or attaching a persona. It does not create a Memory resource. Ordinary sending appends the message and queues a run atomically; note saving needs only working storage.

The composer rejects blank text and limits input to 16,000 characters. Enter uses the current send mode, while Shift+Enter inserts a newline. Normal sending requires Hikari readiness; otherwise the composer offers note saving.

The browser outbox retains an unconfirmed submission's stable message identity, text and send mode. Reload reconciles saved state before retry, and retry must preserve note mode so an uncertain save cannot become an inference request. An optimistic message is not a storage receipt.

Outbox storage is scoped to both agent and conversation; ordinary drafts are scoped to the agent. Selecting a different oldest active conversation after a Tsuchi trash or restore remounts chat state, and late updates from the previous conversation cannot replace the selected history. This keeps a retry attached to its original conversation without discarding the agent's unfinished draft.

Notes remain ordinary user messages in the stored conversation. Later context preparation can include them within the same bounded recent-history window as other messages. Storage persistence and model recall are separate responsibilities.
