# Spaces

Kadan lets users group related chats into **Spaces**, similar to projects in ChatGPT. A Space has a name, optional instructions and its own chats. Kadan provides the Spaces; Tsuchi stores them with the agent's other records. Spaces do not hold files, documents or memories.

## Creating and editing

Choose **+** beside **Spaces** in the sidebar, enter a name and optional instructions, and choose **Create Space**. The new Space opens on its own page. **Space settings** on that page changes the name or instructions. Saving reports an error rather than replacing changes made elsewhere since the page loaded; reopen the Space and try again.

## Chatting in a Space

Typing in **New chat in …** on a Space page starts a chat inside that Space and sends the message once the agent can reply. Every chat in a Space follows its instructions in addition to the agent's persona and shared instructions. Instruction changes apply to replies prepared after the save; work that has already started keeps the instructions it began with.

The Space page lists its chats, newest first. The sidebar shows a few recent chats under each Space; expand or collapse a Space with its arrow. A chat can be moved into a Space, to another Space, or out of its Space from the chat's **…** menu. Moving a chat keeps its messages and changes which instructions apply to later replies.

## Deleting

**Delete Space** asks for confirmation, then moves the Space and its chats to Tsuchi's trash, where they can be restored. If a reply is saved while the Space is being deleted, Kadan tries again; if it still cannot finish, it reports an error, and any chats already moved to trash stay there. Restoring a Space in Tsuchi does not restore its chats automatically; restore the Space first, then its chats.

## Limits

Spaces belong to one agent. Instructions are not a guarantee: the agent follows them as part of its prompt, and very long instructions take up room otherwise used for recent messages. Voice calls do not use Space instructions.

See [Agent conversations](agent-conversations.md) and [Spaces design](../../architecture/kadan/spaces.md).
