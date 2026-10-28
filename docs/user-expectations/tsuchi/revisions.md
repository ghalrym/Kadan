# Revisions

Tsuchi keeps saved revision history for personas, memories, Spaces, slide decks and conversations. Open a saved resource's **History** tab to inspect earlier snapshots. History records saved states; it cannot recover text that was never saved.

**Restore this version** applies an earlier snapshot as a new revision. It keeps the intervening history rather than erasing it. The restored content replaces the current editable contents, so users should save any draft they want to keep before restoring.

Restoration must reject conflicting edits and invalid links. An earlier snapshot can refer to a parent or attachment now in trash; restore those dependencies before trying again. A resource in trash must itself be restored before its contents can be revised. Errors must remain visible without claiming that restoration succeeded.

Restoring a conversation snapshot changes stored messages. It does not repeat inference, computer actions or other work described by those messages. Revision history also does not provide file-content replacement or a file-version browser.

Returning an item from trash is covered separately in [Restore](restore.md).
