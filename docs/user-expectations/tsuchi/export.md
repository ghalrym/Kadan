# Export

Tsuchi's **Export namespace** downloads a ZIP containing the selected agent's stored content. It includes the agent record, resources, revision history, file metadata and original files, retained activity, pet records, agent runs and image jobs. Trashed content is included.

The archive represents saved state at export time. Unsaved editor drafts and changes saved afterward are not included. Other agents, Hikari's models and playground results, and the agent computer's filesystem are outside its scope. Runner assignments are also excluded.

Export can take time as the archive is assembled and downloaded. Missing stored files must stop the export with an error rather than produce a supposedly complete archive. A failed or interrupted download is not a confirmed export; retry and check the resulting archive before relying on it.

The current application has no namespace ZIP import feature. An export provides portable records and files, but not a one-click way to restore an installation or resume its work. See [Backup](backup.md) for preserving the complete Tsuchi store.
