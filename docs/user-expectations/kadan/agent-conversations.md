# Agent conversations

Kadan lets users hatch an agent and return to one continuous chat with it. Creating an agent saves its records in Tsuchi without downloading or running a model. The current interface does not rename or delete agents.

Hatching requires a name and a unique permanent key. Invalid or duplicate keys produce an error without replacing an existing agent. After an uncertain creation, refresh the agent list before trying again. Agents created in Tsuchi appear in Kadan after refreshing; both applications use the same [namespaces](../tsuchi/namespaces.md).

Each agent has its own saved conversation. Drafts are kept separately in the current browser tab and can be lost when the tab closes; they are not saved messages.

Sending saves the message and queues work. Users can keep typing and send more messages while earlier work waits. Replies can be delayed by other work or model loading. The agent can finish work without sending a visible reply; errors and stopped work must remain visible.

**Sending…** means a message or note is awaiting save confirmation. **Delivered** means it was saved in the conversation; it does not confirm that the model has read it. A later reply may come from earlier work.

While response work is pending, Kadan shows whether it is queued or waiting for a response. When Hikari reports a language-model download or load, Kadan shows that state. These indicators refresh periodically and can lag changes. A loaded model does not prove that this prompt is being processed: Kadan does not currently receive that stage, so it shows no prompt-processing claim or estimated percentage. Downloads remain explicit actions in Hikari Settings; a progress label does not start or retry work.

Saved history survives navigation and service recovery. **Show earlier messages** reveals older messages, but only a limited recent portion is available to each response. Image generation can deliver a result later in the same chat.

Users can [download the saved chat as Markdown](downloading-chat.md), including older saved messages without revealing them first.

Each response has a limit of 24 commands and a bounded context for instructions, recent messages and action results. Reaching either limit fails that response visibly rather than continuing indefinitely. Saved history and messages already delivered remain. Sending a new message starts new work and does not guarantee completion of the earlier task. A saved message can also wait for [runner availability](runner-availability.md).

After a send error, **Reload** checks what was saved. **Retry** under **Unsent message** keeps the original submission and persona. Resolve or discard it before sending another message. Closing the tab does not cancel saved work; use [Stopping chat work](stopping-chat-work.md).

See [Conversation architecture](../../architecture/kadan/agent-conversations.md) for ownership, context and recovery.
