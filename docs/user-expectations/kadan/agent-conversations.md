# Agent conversations

Kadan lets users hatch an agent and return to one continuous chat with it. Creating an agent saves its records in Tsuchi without downloading or running a model. Users can [rename agents](renaming-agents.md); deleting agents is not available.

Hatching requires a name and a unique permanent key. Invalid or duplicate keys produce an error without replacing an existing agent. After an uncertain creation, refresh the agent list before trying again. Agents created in Tsuchi appear in Kadan after refreshing; both applications use the same [namespaces](../tsuchi/namespaces.md).

Each agent has its own saved conversation. Drafts are kept separately in the current browser tab and can be lost when the tab closes; they are not saved messages.

Sending saves the message and queues work. Users can keep typing and send more messages while earlier work waits. Replies can be delayed by other work or model loading. The agent can finish work without sending a visible reply; errors and stopped work must remain visible.

Saved history survives navigation and service recovery. **Show earlier messages** reveals older messages, but only a limited recent portion is available to each response. Image generation can deliver a result later in the same chat.

Each response has a limit of 24 commands and a bounded context for instructions, recent messages and action results. Reaching either limit fails that response visibly rather than continuing indefinitely. Saved history and messages already delivered remain. Sending a new message starts new work and does not guarantee completion of the earlier task. A saved message can also wait for [runner availability](runner-availability.md).

After a send error, **Reload** checks what was saved. **Retry** under **Unsent message** keeps the original submission and persona. Resolve or discard it before sending another message. Closing the tab does not cancel saved work; use [Stopping chat work](stopping-chat-work.md).

Trashing or restoring conversations in Tsuchi can change which chat opens after refreshing Kadan. Unsent submissions remain with their original conversation in the current browser tab and reappear if that conversation is reopened. The agent's ordinary draft remains available across this change.

An **Earlier unsent message** has no recorded conversation and cannot be retried safely. Its text remains available to copy into the current draft or discard. Copying preserves existing draft text and does not send anything. Check saved history before submitting it again to avoid a duplicate.

See [Conversation architecture](../../architecture/kadan/agent-conversations.md) for ownership, context and recovery.
