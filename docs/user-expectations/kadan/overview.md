# Kadan user expectations

Kadan is the everyday interface for working with agents. Users can return to saved conversations, ask agents to act on their computer, generate images and browse stored content. Tsuchi keeps durable records; Hikari supplies inference.

Saved work can continue after the browser closes. A queued message does not promise an immediate response, and stopping work does not undo actions already taken. Each feature page describes its waiting, failure and recovery behavior.

| Feature | What to expect |
| --- | --- |
| [Agent conversations](agent-conversations.md) | Hatch agents, send messages and return to saved history within bounded response context. |
| [Reading chat](reading-chat.md) | Keep the current reading position or jump to the latest message. |
| [Finding chat text](finding-chat-text.md) | Find literal text in the saved conversation. |
| [Conversation history](conversation-history.md) | Let the agent search and read older delivered chat within explicit limits. |
| [Conversation activity](conversation-activity.md) | Review queued, failed, stopped and completed work without inferring external success. |
| [Renaming agents](renaming-agents.md) | Change a display name while preserving identity and saved work. |
| [Agent memory](memories.md) | Let agents maintain memories while users inspect saved records. |
| [Chat attachments](chat-attachments.md) | Upload files, attach them to messages or notes and recover uncertain submissions. |
| [Saving chat notes](saving-chat-notes.md) | Save text without requesting a reply, including when inference is unavailable. |
| [Forgetting memories](forgetting-memories.md) | Ask an agent to move an outdated memory to recoverable trash. |
| [Stopping chat work](stopping-chat-work.md) | Cancel response work and image jobs with separate controls. |
| [Personas](personas.md) | Let agents maintain saved response instructions, with optional additional named personas. |
| [Shared agent instructions](agent-instructions.md) | Change operator-maintained instructions for subsequent agent work. |
| [Agent actions](agent-actions.md) | Ask agents to browse, change files and run commands, with explicit limits on recovery. |
| [Coding with Codex](coding.md) | Sign in with ChatGPT once, let the agent run Codex on its computer and track its threads. |
| [Computer](computer.md) | Watch or control the shared computer without pausing the agent. |
| [Images in conversations](image-generation.md) | Continue chatting while images generate and receive saved results later. |
| [Pets](pets.md) | Create, import, preview, repair and explicitly select animated pets. |
| [Voice input](voice-input.md) | Transcribe an utterance into an editable message draft. |
| [Voice calls](voice-calls.md) | Have a separate spoken conversation with interruption and playback controls. |
| [File delivery](file-delivery.md) | Save completed computer files in Workspace and share their download links. |
| [Opening Workspace files](opening-workspace-files.md) | Copy a saved file onto the computer for inspection or processing. |
| [Workspace](workspace.md) | Browse document hierarchies, memories and files; explicitly refresh saved changes. |
| [Service status](service-status.md) | Distinguish storage access, inference readiness and model setup needs. |
| [Runner availability](runner-availability.md) | Inspect and manage runner assignments, with capacity and recovery limits. |

See [Install and use](../install-and-use.md), [Security and operator responsibility](../security.md) and [Performance tracing](../tracing.md) for shared expectations. [Tsuchi](../tsuchi/overview.md) covers storage management, and [Hikari](../hikari/overview.md) covers direct inference.
