# Kadan Server user expectations

Kadan Server runs everything Kadan needs on one computer: conversations, storage and the AI models. Its management pages are part of the same web app as Kadan, at `/server`, so they work in the Kadan Server window and in any browser that can reach the server. They use Kadan's look and plain language; users never need to know the names of the services behind them.

The sidebar offers, from top to bottom:

| Section | What to expect |
| --- | --- |
| [Home](home.md) | Whether everything is ready, the address to connect Kadan, what each ability needs, and memory in use. |
| [AI models](ai-models.md) | Download the model behind each ability, use a Claude or ChatGPT subscription instead, and adjust advanced quality options. |
| [Your agent](your-agent.md) | The agent's memories, persona and its history, chat history, trash and backup. The row shows the agent's name and pet. |
| [Activity](activity.md) | What the AI models have worked on recently, in plain words. |
| [Settings](settings.md) | Addresses for connecting Kadan and other apps, and where to start, stop and update the server. |

In a browser, **Open Kadan** at the bottom of the sidebar opens the Kadan app. Inside the Kadan Server window, where there is nowhere to go back to, that link is hidden.

The old Hikari and Tsuchi pages are gone. Their addresses forward to **AI models** (`/hikari`) and **Your agent** (`/tsuchi`). Kadan Server has no test playgrounds for chat, decisions, images, speech or dictation; Kadan and other apps use those abilities directly.

## The Kadan Server window

The desktop window keeps the controls that only make sense on the server's own computer:

- **Start Server** and **Stop Server** in the header, with the address it is serving on.
- The **Kadan Server** tab shows the pages above while the server runs. While it is stopped or starting, the tab explains that and offers **Start Server**.
- **This computer** shows the connection addresses, background services with restart buttons, hardware, version and updates, and server settings (network access, port and data folder).
- **Logs** shows recent output from the background services.

See [Install and use](../install-and-use.md) for installation, updates and data folder expectations.
