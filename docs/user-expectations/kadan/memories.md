# Agent memory

Kadan agents manage their own durable memories; Tsuchi stores the records. The agent decides what to retain and keeps that saved context useful for its work with the user. Users should be able to inspect what their agent retained without having to maintain its notes. Memories are separate from Workspace documents, which are work products users and agents work on.

During text chat, an agent can list, search, read, create and update its own memories, or move an outdated memory to recoverable Trash. These tools are available without an agent computer. Their use depends on the model choosing them; storage does not provide automatic recall in every response or scheduled background curation. Voice calls have no memory tools.

Users can search and read the selected agent's saved memories in Workspace's separate **Memories** tab. Switching agents shows that agent's records. Choose **Correct memory** to edit its title and Markdown in Kadan. Links, tags, attachments, identity and revision history stay intact; other metadata and revision restoration remain in Tsuchi. **Open Tsuchi** opens that storage editor. **Refresh** checks saved changes. Opening a memory does not add it to the response's context; the agent must search or read the records it needs.

The correction editor reads the current memory before editing. A delayed Workspace refresh does not replace a newer confirmed correction with older text. Unsaved corrections remain in this browser tab when navigating away, separately for each agent and memory; closing the tab can lose them. If tab storage is full or unavailable, keep the editor open until saving or discarding the correction. Discarding an unsaved correction asks for confirmation.

A failed or uncertain correction keeps the draft and requires **Check saved version** before another save. The check shows the saved text for comparison without replacing the correction. If the saved text already matches, no further save is needed. A save does not change context already prepared for a response.

A loading or search failure must remain distinguishable from no memories. Failed saves must be reported, and conflicting edits require another read rather than silently replacing newer content. An interrupted write may already have happened: the agent must inspect the saved record before retrying. Asking an agent to remember something, or waiting for a reply, is not confirmation that it saved a memory.

Agent-managed memory does not promise that every conversation is retained or that every saved memory is recalled. [Forgetting](forgetting-memories.md) removes a memory from active reads while preserving revisions and recovery. It does not erase chat history or already prepared response context. [Stopping chat work](stopping-chat-work.md) cannot undo a write already in progress.

See [Tsuchi memories](../tsuchi/memories.md) for editing, revisions and recovery, and [Memory tools](../../architecture/kadan/memories.md) for the storage boundary.
