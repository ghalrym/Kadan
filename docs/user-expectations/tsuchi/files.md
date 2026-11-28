# Files

Tsuchi stores the agent's files, including images, audio, video, PDFs and general files. Users upload, organize and download them in Kadan's [Space](../kadan/space.md) and attach them in chat. The default upload limit is 100 MiB per file; operators can change it.

Choosing a local file does not upload it. A confirmed upload makes it available for linking to that agent's stored content. Upload and metadata failures must show an error. After an uncertain upload, check Space before uploading again to avoid duplicate records.

Stored file contents cannot be replaced in place. Upload a new file to change the bytes, then update any links that should use it. Editing a title or tags does not change the original filename or contents. Conflicting metadata edits must not silently overwrite newer changes.

Playback depends on the stored format, browser support and whether the file is served for playback or download. Only verified WAV audio is served for inline playback. General files download without an in-app preview. Uploading a PDF or recording does not extract text or transcribe it. Search matches metadata rather than file contents.

Files can be kept in a [folder](folders.md) and pinned through Kadan's [Space](../kadan/space.md); editing a file's details keeps its folder and pin. Files used by active content must be unlinked before moving them to [trash](trash.md). Trash retains their bytes. See [Restore](restore.md) for recovery.
