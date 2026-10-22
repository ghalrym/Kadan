# Renaming agents

Kadan offers **Rename agent** beside **Edit pet** in the selected agent’s navigation. Renaming changes the display name saved by Tsuchi. The permanent key, conversations, drafts, files, persona records and selected pet stay with the same agent. Saving updates its name in Kadan without restarting the chat; other open views pick up the name after refreshing. Names cannot be blank.

Cancel before saving leaves the name unchanged. While saving or reloading, the dialog waits for the result. If saving fails or its outcome is uncertain, the draft stays available and **Reload current name** retrieves the saved name for comparison before another save. A concurrent rename must not be silently overwritten. Closing the browser after saving starts does not undo a name already saved.

See [Conversation architecture](../../architecture/kadan/agent-conversations.md) for storage ownership and conflict handling.
