# File delivery

Kadan agents can find files stored in their [Space](space.md) and share them in chat as download links. Files reach Space when users upload them to chat or Space, for example. Agents cannot create files from scratch, but they can write Space documents.

An agent can list or search its active files by title, filename, tags or source details. It cannot search file contents. It can also read a file's details and download link without loading its contents into the conversation. Missing or trashed files are reported as errors and never recreated. Finding a file does not itself send a message, and a link in chat does not keep the file from being moved to Trash later.

Users manage stored files and Trash in Tsuchi. Refresh Space if it was already open when a file changed.

See [File lookup design](../../architecture/kadan/file-delivery.md).
