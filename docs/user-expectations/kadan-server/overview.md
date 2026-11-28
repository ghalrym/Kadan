# Kadan Server user expectations

Kadan Server is the part of Kadan that runs everything on one computer: conversations, storage and the AI models. Its management pages are part of the same web app as Kadan, at `/server`, so they open in the Kadan app and in any browser that can reach the computer running it. They use Kadan's look and plain language; users never need to know the names of the services behind them.

The sidebar offers, from top to bottom:

| Section | What to expect |
| --- | --- |
| [Home](home.md) | Whether everything is ready, the address to connect Kadan, what each ability needs, and memory in use. |
| [AI models](ai-models.md) | Download the model behind each ability, use a Claude or ChatGPT subscription instead, and adjust advanced quality options. |
| [Your agent](your-agent.md) | The agent's memories, persona and its history, chat history, trash and backup. The row shows the agent's name and pet. |
| [Activity](activity.md) | What the AI models have worked on recently, in plain words. |
| [Settings](settings.md) | Addresses for connecting Kadan and other apps, and, in the Kadan app on the hosting computer, starting, stopping and configuring Kadan's services. |

**Open Kadan** at the bottom of the sidebar returns to the Kadan app. In the Kadan app, Kadan Server opens from **Settings → Kadan Server → Open Kadan Server**.

The old Hikari and Tsuchi pages are gone. Their addresses forward to **AI models** (`/hikari`) and **Your agent** (`/tsuchi`). Kadan Server has no test playgrounds for chat, decisions, images, speech or dictation; Kadan and other apps use those abilities directly.

When you open these pages in the Kadan app on another computer, they show the computer that runs Kadan, not the one you are sitting at.

See [Install and use](../install-and-use.md) for installation, updates and data folder expectations.
