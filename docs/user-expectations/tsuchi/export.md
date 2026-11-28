# Export

**Download backup** in Kadan Server's [Your agent → Backup](../kadan-server/your-agent.md) downloads a ZIP containing the agent's stored content. It includes the agent record, resources, revision history, file metadata and original files, retained activity, pet records, agent runs and image jobs. Trashed content is included.

The archive represents saved state at export time. Unsaved drafts and changes saved afterward are not included. Other agents and Hikari's models are outside its scope. Runner assignments are also excluded.

Export can take time as the archive is assembled and downloaded. Missing stored files must stop the export with an error rather than produce a supposedly complete archive. A failed or interrupted download is not a confirmed export; retry and check the resulting archive before relying on it.

Kadan has no way to import a backup ZIP. An export provides portable records and files, but not a one-click way to restore an installation or resume its work. See [Backup](backup.md) for preserving the complete Tsuchi store.
