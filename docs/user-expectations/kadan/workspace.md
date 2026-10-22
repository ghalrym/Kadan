# Workspace browsing

Kadan's Workspace shows the selected agent's Memories, Documents and Files stored in Tsuchi. Search filters the current collection, and changing agents opens that agent's records. A loading error must be distinguishable from an empty collection.

**Refresh** reloads the collection and open record without changing the search, collection or selection. Documents also refresh their parent path and subdocuments. A record can stay open even when it no longer matches the search. If it is gone, trashed or no longer belongs in that view, Kadan replaces its content with an unavailable notice.

**Retry** repeats a failed read without changing the search or closing the selected record. A temporary error keeps the last loaded record visible with an error. Changing the collection or agent dismisses the selection; results from earlier reads must not replace the current view.

Users can read memories and Spaces, open a Space's children, and download files. Files show their name, type and size; the view does not preview arbitrary file contents.

Workspace is read-only. **Open Tsuchi** opens the storage editor in another tab. Use Refresh to see changes made there or by an agent; Workspace does not update continuously.

Search matches stored text and metadata. It does not search inside uploaded PDFs, recordings or other file contents, so a missing match does not prove that a file lacks the information.

Opening a record does not attach it to chat or make the agent recall it. The current response flow does not automatically retrieve the memories or Spaces being viewed.

See [Workspace access](../../architecture/kadan/workspace.md) for the storage boundary and [Tsuchi memories](../tsuchi/memories.md), [Spaces](../tsuchi/spaces.md), [files](../tsuchi/files.md) and [revisions](../tsuchi/revisions.md) for editing and revisions.
