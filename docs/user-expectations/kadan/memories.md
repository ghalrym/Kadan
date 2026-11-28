# Agent memory

Kadan agents manage their own durable memories; Tsuchi stores the records. The agent decides what to retain and keeps that saved context useful for its work with the user. Memories are separate from saved documents, which are work products users and agents work on.

During text chat, an agent can list, search, read, create and update its own memories, or move an outdated memory to recoverable Trash. In addition, Kadan automatically loads the agent's core memories into each conversation and voice call, recalls relevant saved memories when a message mentions them, and checks completed chat turns in the background after a brief pause in conversation to save important facts the user shared without slowing down active replies (cancelling background work immediately if a new message arrives). Voice calls have no memory tools, but they start with the agent's saved core memories.

Kadan does not show or edit memories. Users can search, correct and forget them in Kadan Server's [Your agent → Memories](../kadan-server/your-agent.md); an edit takes effect on the agent's next response.

A loading or search failure must remain distinguishable from no memories. Failed saves must be reported, and conflicting edits require another read rather than silently replacing newer content. An interrupted write may already have happened: the agent must inspect the saved record before retrying. Asking an agent to remember something, or waiting for a reply, is not confirmation that it saved a memory.

Agent-managed memory does not promise that every conversation is retained or that every saved memory is recalled. [Forgetting](forgetting-memories.md) removes a memory from active reads while preserving revisions and recovery. It does not erase chat history or already prepared response context. [Stopping chat work](stopping-chat-work.md) cannot undo a write already in progress.

See [Tsuchi memories](../tsuchi/memories.md) for editing, revisions and recovery, and [Memory tools](../../architecture/kadan/memories.md) for the storage boundary.
