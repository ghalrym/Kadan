# Trash

Kadan, Kadan Server and the agent can move saved resources and files to trash. Trashed items leave the ordinary active lists and appear in Kadan Server's [Your agent → Trash](../kadan-server/your-agent.md). Restore an item before editing it.

Trash retains content, revision history and file bytes. It does not reclaim storage, and there is no permanent-delete or empty-trash action. Restarting the service does not clear trash.

A Space with active children, or a folder holding active items or files, cannot be trashed until those items are moved or trashed. A file or image used by active content must first be unlinked. These conflicts produce errors; trash does not silently remove the dependencies. Conflicting changes must also be reported.

Trashing artwork used by the selected pet clears its selection. Trashing a conversation removes that chat from Kadan. Moving records to trash is not a control for stopping agent work; use Kadan's [stop controls](../kadan/stopping-chat-work.md).

After a failed or uncertain operation, refresh the list and check the item's state before retrying. See [Restore](restore.md) for returning items to use.
