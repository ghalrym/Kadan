# Navigation and the agent card

Kadan's layout follows coding-assistant apps such as T3 Code and Codex: a sidebar on the left and the selected page filling the rest of the window, with no title bar above it. On narrow screens the sidebar opens from the menu button in the top-left corner.

## Sidebar

From top to bottom, the sidebar offers:

- The Kadan name and logo at the very top.
- **Search threads** and **New thread** below it. Search filters [Work](work.md) threads only; New thread starts a Work thread.
- Each connected external agent's name and pet open your [private chat](direct-chats.md) with it, most recently active first.
- Your agent's name and pet open its single ever-persistent conversation thread. A dot appears on it when the agent posts while you are elsewhere in Kadan; it can take a few seconds to appear and clears when you open the chat.
- **Group Chat** opens the shared [Group Chat](group-chat.md) room for you, Kadan's agent and connected external agents. Its row shows the pets of the people in the room, counts anyone beyond the first few, and shows a dot when there are messages you have not seen. The dot can take a few seconds to appear.
- **Space** opens the [Space](space.md) library of pages, slides and files. Its row previews your three most recently changed items, with thumbnails for images.
- **Plugins** opens the [Plugins](plugins.md) page. Its row shows the icons of your installed plugins, counts any beyond the first three, and shows an amber dot when a turned-on plugin has a connection problem.
- Both previews refresh when Kadan starts and after you leave those pages.
- **Work** lists native [Work](work.md) coding threads the way T3 Code does: active threads first, then collapsible **Working**, **Snoozed**, and **Settled** shelves with counts at the bottom of the list.
- At the bottom, your name and pet open [Your profile](profile.md).
- Below your profile, **Server settings** (in the standalone Kadan app) shows the connected server address and connection status and lets you switch servers, alongside a refresh button that reloads server status, the agent and its chats. Tsuchi and Hikari management live in **Kadan Server**, not inside the Kadan client app.

Lists in the sidebar reflect what Kadan last loaded. Changes made in Kadan Server appear after returning to Kadan or refreshing.

## Agent card

Clicking the agent's pet in chat opens its card, which shows the agent's status and, in order: **Call** to start a [voice call](voice-calls.md), **Rename** to [rename the agent](renaming-agents.md), **Persona** to [edit its persona](personas.md), **Edit pet** to [change its pet](pets.md), and, set apart below a divider, **Delete agent**, which asks you to type `<agent name> Delete` before permanently removing the agent, all its memories, chat history, persona, Space items, and pets so you can start fresh with a new default agent. Each action closes the card. A note at the bottom of the card explains that the agent's memories can be seen in Kadan Server.

See [Agent conversations](agent-conversations.md) for how chats behave.
