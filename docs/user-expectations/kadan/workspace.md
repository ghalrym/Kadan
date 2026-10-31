# Workspace browsing

Kadan provides a browser for the selected agent's content; Tsuchi stores the records. Workspace starts on **Documents**, showing Markdown Spaces and slide decks in a parent/child hierarchy. Memories, personas and conversations stay outside that document hierarchy. **Memories** and **Files** have separate tabs. Users can [save an assistant reply](saving-chat-results.md) as a document in this collection.

Users can open a document's subdocuments, follow breadcrumbs to an ancestor, or use **Back to** to return to its parent. Documents with an unavailable parent remain reachable. Search filters the document list without closing the open document or removing its parent navigation. Opening a result loads its available parent trail.

Returning from chat preserves Workspace selection, search and scroll position for the same agent. This does not promise persistence after reloading the page or switching agents. Changing the collection or agent dismisses the selection; late results from earlier reads must not replace the current view.

**Refresh** reloads the collection and open record without changing the search or selection. Documents also refresh their parent path and subdocuments. A record can remain open when it no longer matches the search. If it is gone, trashed or no longer belongs in that view, Kadan replaces obsolete content with an unavailable notice.

**Retry** repeats a failed read while retaining the search and open record. A temporary failure keeps the last loaded record visible with an error. Loading and failed reads must remain distinct from empty results. Workspace does not continuously synchronize external edits.

Users can search and read saved [memories](memories.md) and download files. Files show their name, type and size; Workspace does not preview arbitrary file contents. Search matches stored text and metadata, without extracting text from uploaded PDFs, recordings or other file contents. A missing match does not prove that a file lacks the information.

Documents and files are read-only in this browser. Users can [correct saved memories](memories.md) in the Memories tab. **Open Tsuchi** opens the storage editor in another tab. During text chat, agents can create Markdown Workspace documents, including subdocuments under an active Space belonging to the same agent. A draft, failed save or uncertain result is not confirmation of a saved document; check Workspace before retrying an uncertain write. Stop cannot undo a write already started. This creation tool does not supply updates to existing documents or persistent project management; those remain [proposals](../proposals.md).

Opening a record does not attach it to chat or make the agent recall it. Current responses do not automatically retrieve displayed documents or memories. Agents can explicitly [save computer files](file-delivery.md) into Workspace and [copy stored files onto the computer](opening-workspace-files.md). Chat attachments remain a [proposed capability](../proposals.md).

See [Workspace access](../../architecture/kadan/workspace.md) for ownership and [Tsuchi Spaces](../tsuchi/spaces.md), [files](../tsuchi/files.md) and [revisions](../tsuchi/revisions.md) for editing and recovery.
