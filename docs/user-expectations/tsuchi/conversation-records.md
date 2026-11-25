# Stored conversations

Tsuchi's Conversations view manages saved records. Creating a conversation or adding a message stores data without asking an agent to reply or running inference. Use Kadan chat to ask an agent to act.

Save the conversation before adding messages. Messages can represent users, assistants, system instructions or tools, and can include text, attachments and links to stored content. Referenced content must belong to the same agent.

Conflicting edits produce an error rather than silently replacing newer changes. Editing a stored message does not repeat the action or inference it describes.

Explicit coding continuations queued by Kadan retain their original task and
conversation with the saved response work. Editing the displayed message does not
change or repeat that queued task.

Tsuchi stores each of an agent's Kadan chats as a conversation. A conversation can belong to a Kadan Space, and its details show that Space as its parent. Trashing a conversation removes the chat from Kadan; restoring it brings the chat back, but only after its Space is restored.

See [Conversation storage](../../architecture/tsuchi/conversations.md) for identifiers and version checks.
