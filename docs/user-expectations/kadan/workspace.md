# Workspace browsing

Kadan provides browsing and navigation for the selected agent's content; Tsuchi stores the records. The proposed experience below is separate from the behavior currently on master.

## Intended experience

The Workspace browsing changes in [draft PR #398](https://github.com/ghalrym/Kadan/pull/398) are not yet merged. They start Workspace on **Documents**, showing existing Spaces and slide decks in a parent/child hierarchy. Memories, personas and conversations stay outside that document hierarchy.

Users can open a document's subdocuments, follow breadcrumbs to an ancestor, or use **Back to** to return to its parent. Documents with an unavailable parent remain reachable in the list.

Search filters the document list without closing the open document or removing its parent navigation. Opening a document from search loads its available parent trail. Returning from chat preserves Workspace selection, search and scroll position for the same agent. This does not promise persistence after a page reload or switching agents.

**Memories** is a separate read-only tab where users can search and view the selected agent's saved memories. It does not include those memories among a document's subdocuments. See [Agent memory](memories.md) for the intended agent-managed memory experience and current recall limits. **Files** remains a separate tab with metadata and downloads.

The proposed browser remains read-only, with editing available through **Open Tsuchi**. Loading and failed reads must remain distinct from empty results. **Retry** repeats the failed collection or subdocument read while retaining the search and any open document. These navigation changes do not add document editing, sharing or automatic recall.

## Current behavior

Kadan's Workspace shows the selected agent's Memories, Spaces and Files stored in Tsuchi. Search filters the current collection, and changing agents opens that agent's records. A loading error must be distinguishable from an empty collection.

Users can read memories and Spaces, open a Space's children, and download files. Files show their name, type and size; the view does not preview arbitrary file contents.

During text chat, an agent can create a Markdown document in Workspace or place a new subdocument under an active Space belonging to that agent. Saving a document does not add editing controls to the browser. Updating existing documents and persistent project planning remain separate proposed work.

The Workspace browser is read-only. **Open Tsuchi** opens the storage editor in another tab. Changes made there may require another search or reopening Workspace to appear.

Search matches stored text and metadata. It does not search inside uploaded PDFs, recordings or other file contents, so a missing match does not prove that a file lacks the information.

Opening a record does not attach it to chat or make the agent recall it. The current response flow does not automatically retrieve the memories or Spaces being viewed.

See [Workspace access](../../architecture/kadan/workspace.md) for the storage boundary and [Tsuchi memories](../tsuchi/memories.md), [Spaces](../tsuchi/spaces.md), [files](../tsuchi/files.md) and [revisions](../tsuchi/revisions.md) for editing and revisions.
