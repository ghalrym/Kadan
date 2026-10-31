# Stored personas

Tsuchi lets users create and edit personas for the selected agent. A persona contains named Markdown instructions and can include tags, source links, attachments and a parent Space. Linked content must belong to the same agent.

The editor offers text editing and preview. Changes persist only after saving; unsaved drafts can be lost on reload. Closing with unsaved edits asks whether to discard them. Failures and conflicting saves must remain visible without silently replacing newer changes.

Saving a persona does not run inference or make a new named record the agent's owned persona. Kadan automatically includes its existing owned persona in later text responses and new voice calls. An additional named persona requires [explicit selection](../kadan/personas.md). Tags, sources and attachments organize the stored record; they do not automatically add attached file contents to the agent's instructions.

Edits in Tsuchi affect the same record Kadan uses. Queued work can receive the edited instructions when it starts; work that already prepared its instructions keeps them when resumed. Trashing the owned persona prevents new work using it until it is restored; choosing another additional persona does not replace the owned one. An unavailable additional selection also causes an error.

Personas support [revision history](revisions.md), [trash](trash.md) and [restore](restore.md). Restoring the same owned persona restores its use. Restoring another named persona does not select it in Kadan.
