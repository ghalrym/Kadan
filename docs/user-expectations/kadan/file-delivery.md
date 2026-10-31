# File delivery

Kadan agents can copy a completed file from their computer into their Workspace Files, then send a download link in chat. Tsuchi stores the copy. Saving a file does not itself send a message, and a chat claim without a confirmed save does not prove the file exists.

Each transfer supports up to 100 MiB; an operator's storage limit can be lower. The computer and Tsuchi must be available. A failed or interrupted transfer must not be reported as a completed save. After an uncertain result, the agent can check the saved file without uploading again. Restart recovery never rereads a potentially changed source to repeat a save automatically.

The original file stays on the shared computer. Later edits to it do not update the saved copy. A new explicit save creates another file. The stored copy can be downloaded from the agent's Workspace or its chat link; refresh the Files view if it was already open when the file was saved.

Kadan checks for cancellation before uploading. Stopping a response cannot undo an upload already in progress. A saved file can remain even if the response stops before delivering its link. Users manage saved files and Trash in Tsuchi. A trashed file is not recreated automatically.

See [Computer](computer.md), [Workspace](workspace.md), and [File transfer design](../../architecture/kadan/file-delivery.md).
