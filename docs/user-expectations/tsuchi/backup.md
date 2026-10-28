# Backup

A complete Tsuchi backup preserves its entire data store, including all agents, saved resources, revision history, trash and uploaded files. It differs from a selected agent's [export](export.md). The current website has no automatic backup scheduler or restore wizard; operators create and keep backups themselves.

Stop Tsuchi and keep it stopped while copying the store so that records and files cannot change partway through the backup. Saving and retrieving content are unavailable during that period, and Kadan work depending on Tsuchi can wait or fail. A partial or unreadable archive is not a usable backup.

Restoration should use a new empty store, preserving the original until the restored copy has been inspected. It recovers the saved state at backup time; later changes and unsaved drafts are absent. Restoring storage does not undo external actions or prove that interrupted work has completed.

Tsuchi backups do not include Hikari's model downloads, the agent computer's files or deployment configuration stored elsewhere. Those need separate preservation when recovering the whole system. Stopping or rebuilding containers preserves Tsuchi's persistent store, but deleting its data volume does not.

The [operator backup and restore procedure](../../architecture/tsuchi/backup.md) describes how to copy and inspect the store.
