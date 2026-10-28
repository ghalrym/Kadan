# Trash

Tsuchi lets users move saved resources and files to trash for the selected agent. Trashed items leave the ordinary active lists and appear in **Trash**, where users can open them for inspection. Restore an item before editing it.

Trash retains content, revision history and file bytes. It does not reclaim storage, and there is no permanent-delete or empty-trash action in the current interface. Restarting the service does not clear trash.

A Space with active children cannot be trashed until those children are moved or trashed. A file or image used by active content must first be unlinked. These conflicts produce errors; trash does not silently remove the dependencies. Conflicting changes must also be reported.

Trashing artwork used by the selected pet clears its selection. Trashing a conversation can change which conversation Kadan opens. Moving records to trash is not a control for stopping agent work; use Kadan's [stop controls](../kadan/stopping-chat-work.md).

After a failed or uncertain operation, refresh the list and check the item's state before retrying. See [Restore](restore.md) for returning items to use.
