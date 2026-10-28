# Forgetting memories

Users can ask a Kadan agent to forget an outdated saved memory. The agent can move that memory to Tsuchi's trash, removing it from active memory lists and ordinary memory reads. Asking is not confirmation: the agent should report success only after storage confirms the change.

Forgetting is reversible. Tsuchi retains the record, its revisions and linked files; users can [restore it from Trash](../tsuchi/restore.md). This does not erase conversation history, information already included in a response's context, or other records containing the same information. It does not reclaim storage.

A newer edit prevents forgetting an older version. The agent must check the current record before trying again. After an interrupted or uncertain operation, it can check whether the memory is now in trash without reading its forgotten contents. Stopping the response cannot undo a change already in progress.

See [Memory tools](../../architecture/kadan/memories.md) for storage and recovery behavior.
