# Agent memory

Kadan agents manage their own durable memories; Tsuchi stores the records. The agent decides what to retain and keeps that saved context useful for its work with the user. Users should be able to inspect what their agent retained without having to maintain its notes. Memories are separate from Workspace documents, which are work products users and agents work on.

During text chat, an agent can list, search, read, create and update its own memories, or move an outdated memory to recoverable Trash. These tools are available without an agent computer. Their use depends on the model choosing them; storage does not provide automatic recall in every response or scheduled background curation. Voice calls have no memory tools.

Users can search and read the selected agent's saved memories in Workspace's separate **Memories** tab. Switching agents shows that agent's records. This view is read-only; **Open Tsuchi** opens the storage editor. **Refresh** checks saved changes. Opening a memory does not add it to the response's context; the agent must search or read the records it needs.

A loading or search failure must remain distinguishable from no memories. Failed saves must be reported, and conflicting edits require another read rather than silently replacing newer content. An interrupted write may already have happened: the agent must inspect the saved record before retrying. Asking an agent to remember something, or waiting for a reply, is not confirmation that it saved a memory.

Agent-managed memory does not promise that every conversation is retained or that every saved memory is recalled. [Forgetting](forgetting-memories.md) removes a memory from active reads while preserving revisions and recovery. It does not erase chat history or already prepared response context. [Stopping chat work](stopping-chat-work.md) cannot undo a write already in progress.

See [Tsuchi memories](../tsuchi/memories.md) for editing, revisions and recovery, and [Memory tools](../../architecture/kadan/memories.md) for the storage boundary.
