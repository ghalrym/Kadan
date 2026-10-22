# Restore

Tsuchi lets users open an item in **Trash** and choose **Restore** to return it to the selected agent's active collection. Restoration retains the item's identity and saved contents. It does not create a separate copy or move content to another agent.

A resource's parent Space and required attachments must be active before it can be restored. Restore parents and files first, then their dependent resources. Restoration does not automatically restore a whole hierarchy or silently discard unavailable links.

An item changed elsewhere can cause a conflict. Failures must remain visible, and an item must not be shown as restored without confirmation. After an uncertain result, refresh and check its state before retrying.

Restoring a file does not automatically select a pet that previously used it. Restoring a conversation can affect which chat Kadan opens, but it does not repeat the actions recorded in that conversation.

Choosing an older saved snapshot is a separate operation covered in [Revisions](revisions.md). Returning an item from trash keeps its current saved contents.
