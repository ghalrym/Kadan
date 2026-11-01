# Chat attachments

Kadan lets users attach files to a message or saved note. Choose files beside the composer, wait for each upload to be confirmed, then send the message. A message can include up to ten files totaling 100 MiB. Tsuchi stores the files, and an operator's storage limit may be lower.

Uploading and sending are separate. Uploading saves a file into the agent's Workspace; sending associates it with the message. Removing an attachment from the draft does not delete its stored file. Notes can contain attachments even when inference or the agent's computer is unavailable. A nonblank text message is still required.

Each attachment shows whether it is waiting, uploading, checking a previous upload, ready or needs attention. An uncertain upload must be checked before retrying. After a reload, Kadan can recover a confirmed upload by its saved identity. If the file was not saved and the browser no longer has its bytes, users must reselect the original file or remove the attachment.

Attachment drafts belong to the conversation where they were selected. A failed send retains its original attachment list alongside its message, persona and reply-or-note choice. Retrying sends that same list even if a newer draft has other attachments. Switching agents or conversations must not attach late upload results to the new conversation.

An agent receives the attached file references and can copy a stored file onto its computer for processing. This does not guarantee that it can interpret every document, recording or image format. A saved attachment is not automatically inserted into the model's text context. Files already in Tsuchi Trash or belonging to another agent cannot be attached; a failed submission must keep its recovery state.

Users can download attachments from chat or find them in Workspace Files. Trashing a file in Tsuchi can make an earlier attachment unavailable. See [File delivery](file-delivery.md) for files produced by agents and [Saving chat notes](saving-chat-notes.md) for notes without a reply.

The Vision subscription backend on this branch does not automatically send attached images for analysis or insert their pixels into a text conversation. It serves the existing screenshot observation capability; attachment handling is unchanged.
