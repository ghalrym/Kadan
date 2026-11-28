# Space

**Space** is Kadan's library for the things you and your agent make and save, modeled on ChatGPT's Space. It holds pages, slide decks, images and other files, organized into folders. Kadan provides the library; Tsuchi stores the items. Chats, memories and personas are not Space items.

## What appears in Space

- **Pages** are editable Markdown documents. A page can contain subpages, which show beneath it.
- **Slides** are slide decks the agent or Tsuchi created.
- **Images** and **Files** are anything uploaded to chat or uploaded to Space.
- **Folders** group any of these, and can hold other folders.

Replies saved with **Save to Space** in chat become pages, and pages the agent writes during a chat appear here too.

## Browsing

**Your items** shows the top level of Space; opening a folder shows what it holds, with a path back to the top. **All**, **Pages**, **Slides**, **Images** and **Files** filter by type across every folder, and search matches item names across every folder. **Pinned** collects pinned items, and **Recents** lists recently changed items. Items can be shown as a list or a grid; the choice is remembered in this browser.

Space shows what Kadan last loaded. Changes made by the agent or in Kadan Server appear after choosing refresh or reopening Space. A failed load is reported with **Retry** and is never shown as an empty Space.

## Creating and changing items

**Create** offers a new **Page**, a new **Folder**, or **Upload files** into the folder being viewed. Uploads are limited to 100 MB per file. Each item's **…** menu can open, rename, pin or unpin, move it to another folder or back to Your items, download files and images, or move it to Trash. A folder cannot be moved into itself or into a folder inside it.

Opening a page shows it formatted; **Edit** changes its title and text. Saving reports an error rather than replacing changes made elsewhere since it was opened; the unsaved text stays in the editor so it can be copied before reopening. Slides open in a viewer with previous and next controls. Images open in a preview; other files show their details with **Download**.

## Trash

**Move to trash** asks for confirmation. A folder moves to Trash with everything inside it. Items in **Trash** can be restored. A page inside a trashed folder or page can only be restored after its parent; a file whose folder is still in Trash is restored to Your items. A file attached to a chat or page cannot be moved to Trash while it is in use; when its folder is trashed, the file moves to Your items instead. Trash is not emptied automatically, and Kadan offers no permanent delete.

## Limits

Space belongs to one agent and has no sharing. Kadan does not continuously sync items, edit slides, preview arbitrary file contents, or read text inside uploaded PDFs or recordings. Opening an item does not add it to the agent's next reply; the agent uses its tools to find and read Space items when mentioned. Agents can [find stored files and share their links](file-delivery.md).

See [Space design](../../architecture/kadan/space.md).
