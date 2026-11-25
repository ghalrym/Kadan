# Agent conversations

Kadan lets users hatch an agent, hold as many chats with it as they like, and return to any of them from the sidebar. Creating an agent saves its records in Tsuchi without downloading or running a model. Users can [rename agents](renaming-agents.md); deleting agents is not available.

Hatching requires a name and a unique permanent key. Invalid or duplicate keys produce an error without replacing an existing agent. After an uncertain creation, refresh the agent list before trying again. Kadan opens the first agent returned by Tsuchi after refreshing; other existing agents are not selectable in Kadan. Both applications use the same [namespaces](../tsuchi/namespaces.md).

## Chats

Each agent has its own saved chats; Tsuchi stores them as conversations. **New chat** starts one. Kadan opens the most recently active chat when it loads, and creates a first chat when the agent has none. An empty chat appears in the sidebar only while it is open; it is listed once it has a message.

A chat without its own name is titled from its first message. **Rename** in the chat's **…** menu gives it a name; renaming keeps its messages, even while a reply is being saved. **Delete** asks for confirmation and moves the chat to Tsuchi's trash, where it can be restored; a reply still in progress cannot be saved into a deleted chat. After the open chat is deleted, Kadan opens the next most recent chat, or a new one.

Each reply receives only the messages of its own chat. Replies, activity, images and background updates stay in the chat that requested them, so switching chats does not move or cancel that work.

One unsent draft is kept per agent in the current browser tab and follows you between chats; it can be lost when the tab closes and is not a saved message. When refresh opens a different agent, Kadan waits for that agent’s chats and personas rather than showing the previous agent’s. If loading fails, it reports the error; retry refreshes the records. A failed refresh for the same agent leaves its last loaded chat visible.

Users can [upload files and attach them](chat-attachments.md) to a message or note, and select saved [page or memory references](request-references.md) for one response. Upload confirmation is separate from message delivery. Sending saves the message and queues work. Users can keep typing and send more messages while earlier work waits. Replies can be delayed by other work or model loading. The agent can finish work without sending a visible reply; errors and stopped work must remain visible.

Each queued reply receives the conversation through the message that requested it. Kadan preserves the text people and agents wrote, including indentation. Message identifiers and attached file references stay separate from that text. Background updates are identified as results from the service that produced them. Earlier saved replies retain their original wording and can be edited in Tsuchi’s Conversations view. If Kadan cannot identify an action in an older saved response after an upgrade, that response reports a failure before repeating any work. Messages already delivered remain available.

The standard deployment uses Inkling 2 Mini for text conversations (or the subscription selected for Language in Hikari Settings). Its checkpoint must be downloaded through Hikari Settings first when using Local; an unavailable worker or incomplete download produces an error without switching models. A completed plain answer becomes an agent reply when the agent is permitted to reply. Empty output can still finish silently. Malformed tool output must fail visibly instead of appearing as a reply.

**Sending…** means a message or note is awaiting save confirmation. **Delivered** means it was saved in the conversation; it does not confirm that the model has read it. A later reply may come from earlier work.

While response work is pending, Kadan shows whether it is queued or waiting for a response. When Hikari reports a language-model download or load, Kadan shows that state. As the agent writes a direct reply, Kadan can show the reply live as text arrives before the final message is saved; this live preview is temporary and clears if the agent takes another action instead of replying, or is replaced by the saved reply once delivered. These indicators refresh periodically and can lag changes. A loaded model does not prove that this prompt is being processed: Kadan does not currently receive that stage, so it shows no prompt-processing claim or estimated percentage. Downloads remain explicit actions in Hikari Settings; a progress label does not start or retry work.

Saved history survives navigation and service recovery. **Show earlier messages** reveals older messages. [Find in conversation](finding-chat-text.md) searches saved text, and [reading controls](reading-chat.md) let users keep their place as replies arrive. Only a limited recent portion is included automatically in each response; the agent can [search and read older delivered chat](conversation-history.md) when needed. Image generation can deliver a result later in the same chat.

Users can [download the saved chat as Markdown](downloading-chat.md), including older saved messages without revealing them first.

Each response has a limit of 24 commands and a bounded context for instructions, recent messages and action results. Reaching either limit fails that response visibly rather than continuing indefinitely. Saved history and messages already delivered remain. Sending a new message starts new work and does not guarantee completion of the earlier task. A saved message can also wait for [runner availability](runner-availability.md).

The agent reasons privately before each step; its reasoning is never shown or saved. Kadan limits private reasoning and reply generation separately so reasoning cannot continue indefinitely. Kadan leaves estimated space for both budgets when selecting recent history and checking saved work. Instructions and tool definitions still have to fit; insufficient space fails visibly before inference. These budgets do not guarantee a fixed response time or that every command within the command-size limit will fit. If Hikari reaches the generation or context limit before completing an agent response, Kadan reports that limit and does not execute a command from the incomplete response. Actions completed earlier in the run remain completed.

After a send error, **Reload** checks what was saved. **Retry** under **Unsent message** keeps the original submission, attachment list, persona and reference versions. Resolve or discard it before sending another message. Closing the tab does not cancel saved work; use [Stopping chat work](stopping-chat-work.md).

Trashing or restoring conversations in Tsuchi can change which chat opens after refreshing Kadan. Unsent submissions remain with their original conversation in the current browser tab and reappear if that conversation is reopened. The agent's ordinary draft remains available across this change.

An **Earlier unsent message** has no recorded conversation and cannot be retried safely. Its text remains available to copy into the current draft or discard. Copying preserves existing draft text and does not send anything. Check saved history before submitting it again to avoid a duplicate.

[Activity](conversation-activity.md) on the agent's card distinguishes pending work, failures, stops and recorded message delivery. Finishing a response does not establish that an external action succeeded.

See [Conversation architecture](../../architecture/kadan/agent-conversations.md) for ownership, context and recovery.
