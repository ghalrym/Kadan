# Spaces

Tsuchi Spaces are Markdown documents that can also contain other stored resources. Users create and edit them as pages in Kadan's [Space](../kadan/space.md).

Parents and children must belong to the same agent. A Space cannot contain itself or one of its ancestors. Moving a resource changes its parent.

Kadan can [save a chat reply](../kadan/saving-chat-results.md) as an independent Space document belonging to the same agent. Repeated saves of that reply preserve later document edits instead of replacing them.

Spaces support tags, sources, attachments and text search. Saving must report failures and conflicting edits. Browsing or saving a Space does not ask an agent to act or automatically add its contents to chat context.

Trashing a Space does not automatically trash its children. Active children must first be moved or trashed. Restore a parent before restoring its children, and restore any required attached files first. Invalid links produce an error rather than a partially restored document.

A Space can also sit in a [folder](folders.md). Kadan shows Space documents as pages in its [Space](../kadan/space.md) library.

See [Revisions](revisions.md), [Trash](trash.md) and [Restore](restore.md) for recovery behavior.
