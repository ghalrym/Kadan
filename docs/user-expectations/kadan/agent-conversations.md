# Agent conversations

Hatch an agent with a name and a permanent key. Kadan suggests a key from the name; it must be unique and contain only lowercase letters, digits, hyphens and underscores, starting with a letter or digit, with at most 64 characters. Creation saves the agent in Tsuchi without downloading or running a model. The current interface does not rename or delete agents.

Each agent has one continuous chat, not a choice of threads. Selecting another agent opens that agent's saved conversation and personas. The browser remembers the selected agent locally. Draft text is kept separately for each agent in the current browser tab's session; a draft is not a saved message or a backup.

Sending a message saves it and queues a response. Users can continue typing and send further messages while work is pending. Runs for one conversation are handled in creation order, one at a time. Queue acceptance does not promise an immediate reply: runner assignment, model loading and Hikari's inference queue can delay it. Initial model loading appears as the agent waking up.

Saved conversation history survives navigation and service recovery. The current response uses a bounded recent window, with at most 97 user and assistant messages and a model-dependent size limit. Older messages remain stored, but they are not all supplied to every response. The interface initially shows recent messages; Show earlier messages reveals more saved history.

The current prototype can finish work without sending a visible reply. Only messages explicitly delivered by the agent appear in chat; completion alone does not produce an answer. Errors and stopped work must remain visible. Image generation runs separately and can deliver its result later in the same chat.

After a send error, Reload checks what was actually saved. Retry under Unsent message preserves the original submission, including its message identity, send mode and persona, rather than creating a new request. Resolve or discard an unconfirmed submission before sending another. Closing a tab can lose browser-only drafts; it does not cancel saved work. See [Stopping chat work](stopping-chat-work.md) for cancellation and the separate image control.
