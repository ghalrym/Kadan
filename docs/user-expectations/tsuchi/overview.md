# Tsuchi user expectations

Tsuchi stores each agent's personas, memories, documents, slide decks, files and conversations. It has no pages of its own: Kadan Server's [Your agent](../kadan-server/your-agent.md) shows memories, the persona and its history, chat history, trash and backups, and Kadan's [Space](../kadan/space.md) edits pages, slides, folders and files. The pages below describe how stored content behaves wherever it is shown.

Saving content does not run an agent or guarantee future recall. Confirmed saves persist in the data store; unsaved drafts do not. Edits must report conflicts, and trash retains data rather than reclaiming disk space.

| Feature | What to expect |
| --- | --- |
| [Personas](personas.md) | Inspect owned and named persona instructions, with revisions and recovery. |
| [Memories](memories.md) | Create and search saved notes without a promise of automatic agent recall. |
| [Spaces](spaces.md) | Edit documents and organize resources through parent and child relationships. |
| [Folders and pins](folders.md) | Keep Kadan's Space folders and pins; a folder holds documents, decks, files and other folders. |
| [Slide decks](slide-decks.md) | Edit slides, layouts, notes and linked images within the supported deck format. |
| [Files](files.md) | Upload, describe, attach and download stored files. |
| [Stored conversations](conversation-records.md) | Manage saved messages without triggering agent responses or repeating actions. |
| [Revisions](revisions.md) | Inspect saved snapshots and restore earlier contents as a new revision. |
| [Trash](trash.md) | Remove items from active collections while retaining their content and bytes. |
| [Restore](restore.md) | Return trashed items after restoring required parents and attachments. |
| [Export](export.md) | Download one agent's portable records and files. |
| [Backup](backup.md) | Preserve and recover the complete Tsuchi store through operator-managed backups. |

See [Install and use](../install-and-use.md), [Security and operator responsibility](../security.md) and [Performance tracing](../tracing.md) for shared expectations. [Kadan](../kadan/overview.md) runs agent conversations and actions; [Hikari](../hikari/overview.md) supplies inference.
