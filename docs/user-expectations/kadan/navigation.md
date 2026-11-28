# Navigation and the agent card

Kadan's layout follows coding-assistant apps such as T3 Code and Codex: a sidebar on the left and the selected page filling the rest of the window, with no title bar above it. On narrow screens the sidebar opens from the menu button in the top-left corner.

## Sidebar

From top to bottom, the sidebar offers:

- The Kadan name and logo at the very top.
- **Search** and **New thread** below it. Search filters [Work](work.md) threads; New thread starts a Work thread.
- The agent's name and pet open its single ever-persistent conversation thread.
- **Group Chat** opens the shared [Group Chat](group-chat.md) room for you, Kadan's agent and connected external agents.
- **Space** opens the [Space](space.md) library of pages, slides and files.
- **Plugins** opens the [Plugins](plugins.md) page.
- **Work** lists native [Work](work.md) coding threads the way T3 Code does: active threads first, then collapsible **Working**, **Snoozed**, and **Settled** shelves with counts at the bottom of the list.
- At the bottom, **Server settings** (in the standalone Kadan app) shows the connected server address and connection status and lets you switch servers, alongside a refresh button that reloads server status, the agent and its chats. Tsuchi and Hikari management live in **Kadan Server**, not inside the Kadan client app.

Lists in the sidebar reflect what Kadan last loaded. Changes made on Tsuchi's pages appear after returning to Kadan or refreshing.

## Agent card

Clicking the agent's pet in chat opens its card, which shows the agent's status and, in order: **Call** to start a [voice call](voice-calls.md), **Rename** to [rename the agent](renaming-agents.md), **Persona** to [edit its persona](personas.md), **Edit pet** to [change its pet](pets.md), and, set apart below a divider, **Delete agent**, which asks you to type `<agent name> Delete` before permanently removing the agent, all its memories, chat history, persona, Space items, and pets so you can start fresh with a new default agent. Each action closes the card.

See [Agent conversations](agent-conversations.md) for how chats behave.
