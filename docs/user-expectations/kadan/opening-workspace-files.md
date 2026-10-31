# Opening Workspace files

Kadan agents can copy a saved Workspace file onto their computer to inspect or work on it. Tsuchi keeps the saved original. The agent needs the file's saved identity; browsing a file does not automatically open it or put its contents into chat.

Each file can be up to 100 MiB. Kadan checks that the file belongs to the current agent, is outside Trash, and arrives unchanged before confirming the computer copy. The computer and Tsuchi must be available. Missing files, unavailable services and incomplete transfers produce errors rather than a usable-file claim.

Each new copy uses a separate temporary path on the shared computer. Editing the copy does not change the original in Workspace. The agent can [save a completed result](file-delivery.md) as another stored file. Computer copies are temporary and may disappear when that environment is recreated; the original stays in Tsuchi.

If Kadan finds changed content at a copy's path, it stops instead of replacing it. The shared computer cannot guarantee that another writer will not change that path between the check and the initial copy. Restart recovery checks that same path and never repeats an uncertain upload. If a copy cannot be confirmed, the agent must inspect the reported path before requesting another copy. Stopping a response prevents writes that have not started, but cannot undo an upload already in progress; a computer copy can remain after cancellation.

This does not add a file viewer or automatically interpret PDFs, spreadsheets or other formats. The agent uses its existing file and computer tools after copying. See [Computer](computer.md) and [File transfer design](../../architecture/kadan/file-delivery.md).
