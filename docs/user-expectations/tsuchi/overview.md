# Tsuchi user expectations

Tsuchi stores each agent's personas, memories, documents, slide decks, files and conversations. Its management interface lets users edit saved content, inspect history, recover trashed items and export records. Kadan uses the same stored content.

Saving content does not run an agent or guarantee future recall. Confirmed saves persist in the data store; unsaved drafts do not. Edits must report conflicts, and trash retains data rather than reclaiming disk space.

| Feature | What to expect |
| --- | --- |
| [Agent namespaces](namespaces.md) | Create and switch between agents with separate stored content. |
| [Personas](personas.md) | Store response instructions and metadata without automatically selecting them in Kadan. |
| [Memories](memories.md) | Create and search saved notes without a promise of automatic agent recall. |
| [Spaces](spaces.md) | Edit documents and organize resources through parent and child relationships. |
| [Slide decks](slide-decks.md) | Edit slides, layouts, notes and linked images within the supported deck format. |
| [Files](files.md) | Upload, describe, attach and download stored files. |
| [Images](images.md) | Browse verified image previews and reuse artwork in stored content. |
| [Stored conversations](conversation-records.md) | Manage saved messages without triggering agent responses or repeating actions. |
| [Search](search.md) | Find active resources and file metadata within the selected agent. |
| [Revisions](revisions.md) | Inspect saved snapshots and restore earlier contents as a new revision. |
| [Trash](trash.md) | Remove items from active collections while retaining their content and bytes. |
| [Restore](restore.md) | Return trashed items after restoring required parents and attachments. |
| [Export](export.md) | Download one agent's portable records and files. |
| [Backup](backup.md) | Preserve and recover the complete Tsuchi store through operator-managed backups. |
| [Storage overview](storage-overview.md) | Inspect counts, retained file bytes and shared database size. |
| [Storage activity](activity.md) | Review bounded recent storage changes. |

See [Install and use](../install-and-use.md), [Security and operator responsibility](../security.md) and [Performance tracing](../tracing.md) for shared expectations. [Kadan](../kadan/overview.md) runs agent conversations and actions; [Hikari](../hikari/overview.md) supplies inference.
