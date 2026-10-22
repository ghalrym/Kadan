# Stored personas

Tsuchi lets users create and edit personas for the selected agent. A persona contains named Markdown instructions and can include tags, source links, attachments and a parent Space. Linked content must belong to the same agent.

The editor offers text editing and preview. Changes persist only after saving; unsaved drafts can be lost on reload. Closing with unsaved edits asks whether to discard them. Failures and conflicting saves must remain visible without silently replacing newer changes.

Saving a persona does not run inference or select it for the next response. Kadan's [persona selection](../kadan/personas.md) determines which saved instructions a response uses. Tags, sources and attachments organize the stored record; they do not automatically add attached file contents to the agent's instructions.

Edits in Tsuchi affect the same record Kadan uses. Queued work can receive the edited instructions when it starts; work that already prepared its instructions keeps them when resumed. Trashing a selected persona can prevent work from starting, so choose another persona when needed.

Personas support [revision history](revisions.md), [trash](trash.md) and [restore](restore.md). Restoring a record does not select it in Kadan.
