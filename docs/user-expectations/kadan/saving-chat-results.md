# Saving replies to Workspace

Kadan offers **Save to Workspace** on saved assistant replies containing text. Enter a document title and choose **Save document**. Tsuchi saves an editable Markdown document in the same agent's Workspace **Documents** collection, with the stored reply's text, source links and attachment references. Files are linked rather than copied. Saving requires Kadan and Tsuchi, without Hikari, a downloaded model or another agent response.

One document is saved per reply. Saving again returns that document with any later edits intact, including its original or edited title. The document and chat are independent: later changes to either do not update the other. General document editing, revisions and restoration remain in Tsuchi.

While saving, confirmation waits for storage. A failure retains the entered title while the form stays mounted and reports that saving was not confirmed. **Retry save** checks or creates the same document, so a lost response does not create a second copy or overwrite edits. Navigating away or reloading can lose the form's title; reopening the reply and saving again still uses the same document identity.

The source conversation and reply must still exist in the same agent's active chat records, including when retrying. A saved document in Trash must be restored in Tsuchi before saving that reply again; Kadan does not silently recreate it. Trashing or changing the source does not remove an already saved document.

Saving does not schedule work, retain every reply automatically, add the document to future response context or guarantee recall. See [Workspace browsing](workspace.md), [Tsuchi Spaces](../tsuchi/spaces.md) and [revisions](../tsuchi/revisions.md).
