# Agent memory

Kadan is responsible for how agents manage and use their memory. Tsuchi stores the saved records. Agents can maintain memories during text conversations; the browser lets users inspect what they have saved.

## Intended experience

Each Kadan agent manages its own memory. The agent decides what to retain and keeps that saved context useful for its work with the user. Users should not need to maintain a collection of notes on the agent's behalf.

Memories is a place for users to view what the selected agent has made and retained. It is separate from Workspace documents, which are work products users and agents work on. Switching agents should show that agent's memories.

Agent-managed memory does not promise that every conversation is retained or that every saved memory is recalled in every response. This expectation does not establish automatic background curation or a schedule for updates.

## Current behavior

During text chat, an agent can list, search, read, create and update its own saved memories. It can also move an outdated memory to Tsuchi Trash when asked to forget it. These actions depend on the model choosing the available tools; they do not establish automatic background curation or retrieval on every response. Voice calls do not have these tools.

Kadan currently shows memories inside [Workspace](workspace.md). Users can search and read the selected agent's saved memories there. The view is read-only; **Open Tsuchi** opens the storage editor. Changes made there may require another search or reopening Workspace to appear.

A loading or search error must be distinguishable from an empty collection. An unavailable view does not establish that memories are missing. Conflicting updates must not silently replace a newer edit. An interrupted save may already have happened, so the agent must check the saved record before retrying. Waiting for a reply is not confirmation that a memory was saved.

Browsing Workspace does not load memories into the agent's response. The agent must search or read the saved records it needs. Forgetting removes a memory from active reads but does not erase chat history, prepared response context or stored revisions. Users can restore it in Tsuchi. [Stopping chat work](stopping-chat-work.md) does not undo a write already in progress.

See [Tsuchi memories](../tsuchi/memories.md) for the current storage editor, search, revisions and trash behavior.
