# Agent memory

Kadan is responsible for how agents manage and use their memory. Tsuchi stores the saved records. The intended experience below is separate from the currently implemented browsing and storage behavior.

## Intended experience

Each Kadan agent manages its own memory. The agent decides what to retain and keeps that saved context useful for its work with the user. Users should not need to maintain a collection of notes on the agent's behalf.

Memories is a place for users to view what the selected agent has made and retained. It is separate from Workspace documents, which are work products users and agents work on. Switching agents should show that agent's memories.

Agent-managed memory does not promise that every conversation is retained or that every saved memory is recalled in every response. This expectation does not establish automatic background curation or a schedule for updates.

## Current behavior

Kadan currently shows memories inside [Workspace](workspace.md). Users can search and read the selected agent's saved memories there. The view is read-only; **Open Tsuchi** opens the storage editor. Changes made there may require another search or reopening Workspace to appear.

A loading or search error must be distinguishable from an empty collection. An unavailable view does not establish that memories are missing. Tsuchi reports save failures and conflicting edits; a draft or failed save is not a saved memory.

Saving or opening a memory does not make the agent recall it. Kadan's current response flow does not automatically retrieve memories, and it does not yet provide the agent-managed memory experience described above. Waiting for a reply is not confirmation that a memory was saved. [Stopping chat work](stopping-chat-work.md) does not undo actions already taken.

See [Tsuchi memories](../tsuchi/memories.md) for the current storage editor, search, revisions and trash behavior.
