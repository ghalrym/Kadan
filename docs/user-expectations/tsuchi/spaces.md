# Spaces

Tsuchi Spaces are editable Markdown documents that can also contain other stored resources. Users can create a Space, preview its text and choose a **Parent Space** under **Details** to nest or move it. The Spaces view shows the hierarchy and opens its documents for editing.

Parents and children must belong to the same agent. A Space cannot contain itself or one of its ancestors. Moving a resource changes its parent; the current interface does not offer drag-and-drop organization.

Spaces support tags, sources, attachments and text search. Unsaved edits are drafts and can be lost on reload. Saving must report failures and conflicting edits. Browsing or saving a Space does not ask an agent to act or automatically add its contents to chat context.

Trashing a Space does not automatically trash its children. Active children must first be moved or trashed. Restore a parent before restoring its children, and restore any required attached files first. Invalid links produce an error rather than a partially restored document.

See [Revisions](revisions.md), [Trash](trash.md) and [Restore](restore.md) for recovery behavior.
