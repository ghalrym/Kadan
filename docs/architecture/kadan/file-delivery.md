# File delivery

Agents deliver stored files by reading Tsuchi asset metadata and sharing the existing Kadan asset-download URL. Kadan never streams file bytes into model context and has no separate execution environment to copy files to or from. Tsuchi remains the only owner of file contents.

## Reading a stored file

`read_workspace_file(file_id)` is a read-only conversation tool scoped to the run's agent. `WorkspaceFiles.read` fetches the asset from Tsuchi and checks the returned namespace, ID and active state before returning metadata plus the download URL. Missing or trashed files produce a tool error; nothing is recreated.

## Discovering stored files

`list_workspace_files(query="", limit=20, offset=0)` is a read-only conversation tool scoped to the run's agent. It accepts limits from 1 to 50, nonnegative offsets and an optional query of at most 500 characters. An omitted or blank query lists all active files. It calls the existing Tsuchi asset query with `trash=false`, requesting one extra match to determine continuation. No file bytes are involved.

Tsuchi owns [metadata matching and pagination](../tsuchi/file-queries.md). Kadan uses that stable ordering without deduplicating display names. A late page or no matches can still scan the collection.

The tool returns `files` and `next_offset`. Each file includes `file_id`, title, filename, media type, byte size, checksum, metadata version, created/updated timestamps and its download URL. Names are display metadata and never deduplicated. A non-null `next_offset` continues with the same query and limit; an empty list with null continuation is a successful no-match result. Pages are separate reads, not a snapshot, so concurrent edits can shift offsets. Kadan validates every returned row, including the lookahead, for active same-agent metadata and rejects malformed, oversized or duplicate-ID lists without returning partial results. Storage failures produce a tool error without `files` or continuation. The read tool rechecks current metadata by the selected stable ID.

The additional tool schemas count against the existing context budget; see [model-facing tool context](tool-context.md). Discovery does not relax admission or automatically change model settings.

See [File delivery expectations](../../user-expectations/kadan/file-delivery.md).
