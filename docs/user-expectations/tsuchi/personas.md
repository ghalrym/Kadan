# Stored personas

Tsuchi stores the agent's persona, which users edit from the agent card in Kadan or from [Your agent → Persona](../kadan-server/your-agent.md) in Kadan Server. A persona contains named Markdown instructions and can include tags, source links, attachments and a parent Space. Linked content must belong to the same agent.

Changes persist only after saving; unsaved text can be lost on reload. Failures and conflicting saves must remain visible without silently replacing newer changes.

Saving a persona does not run inference or make a new named record the agent's owned persona. Kadan automatically includes its existing owned persona in later text responses and new voice calls. An additional named persona requires [explicit selection](../kadan/personas.md). Tags, sources and attachments organize the stored record; they do not automatically add attached file contents to the agent's instructions.

Edits in Kadan Server affect the same record Kadan uses. Queued work can receive the edited instructions when it starts; work that already prepared its instructions keeps them when resumed. Trashing the owned persona prevents new work using it until it is restored; choosing another additional persona does not replace the owned one. An unavailable additional selection also causes an error.

Personas support [revision history](revisions.md), [trash](trash.md) and [restore](restore.md). Restoring the same owned persona restores its use. Restoring another named persona does not select it in Kadan.
