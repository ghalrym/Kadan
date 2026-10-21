# Agent conversations

Kadan lets users hatch an agent and return to one continuous chat with it. Creating an agent saves its records in Tsuchi without downloading or running a model. The current interface does not rename or delete agents.

Each agent has its own saved conversation. Drafts are kept separately in the current browser tab and can be lost when the tab closes; they are not saved messages.

Sending saves the message and queues work. Users can keep typing and send more messages while earlier work waits. Replies can be delayed by other work or model loading. The agent can finish work without sending a visible reply; errors and stopped work must remain visible.

Saved history survives navigation and service recovery. **Show earlier messages** reveals older messages, but only a limited recent portion is available to each response. Image generation can deliver a result later in the same chat.

After a send error, **Reload** checks what was saved. **Retry** under **Unsent message** keeps the original submission and persona. Resolve or discard it before sending another message. Closing the tab does not cancel saved work; use [Stopping chat work](stopping-chat-work.md).

See [Conversation architecture](../../architecture/kadan/agent-conversations.md) for ownership, context and recovery.
