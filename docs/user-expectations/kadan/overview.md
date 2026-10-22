# Kadan user expectations

Kadan is the everyday interface for working with agents. Users can return to saved conversations, ask agents to act on their computer, generate images and browse stored content. Tsuchi keeps durable records; Hikari supplies inference.

Saved work can continue after the browser closes. A queued message does not promise an immediate response, and stopping work does not undo actions already taken. Each feature page describes its waiting, failure and recovery behavior.

| Feature | What to expect |
| --- | --- |
| [Agent conversations](agent-conversations.md) | Hatch agents, send messages and return to saved history within bounded response context. |
| [Renaming agents](renaming-agents.md) | Change a display name while keeping its permanent key and saved content. |
| [Saving chat notes](saving-chat-notes.md) | Save text without requesting a reply, including when inference is unavailable. |
| [Stopping chat work](stopping-chat-work.md) | Cancel response work and image jobs with separate controls. |
| [Personas](personas.md) | Save and select instructions for an agent's responses. |
| [Shared agent instructions](agent-instructions.md) | Change operator-maintained instructions for subsequent agent work. |
| [Agent actions](agent-actions.md) | Ask agents to browse, change files and run commands, with explicit limits on recovery. |
| [Computer](computer.md) | Watch or control the shared computer without pausing the agent. |
| [Images in conversations](image-generation.md) | Continue chatting while images generate and receive saved results later. |
| [Pets](pets.md) | Create, import, preview, repair and explicitly select animated pets. |
| [Voice input](voice-input.md) | Transcribe an utterance into an editable message draft. |
| [Voice calls](voice-calls.md) | Have a separate spoken conversation with interruption and playback controls. |
| [Workspace](workspace.md) | Browse memories, Spaces and files without automatically adding them to chat context. |
| [Service status](service-status.md) | Distinguish storage access, inference readiness and model setup needs. |
| [Runner availability](runner-availability.md) | Understand assignment capacity, queued work and recovery after a runner stops. |

See [Install and use](../install-and-use.md), [Security and operator responsibility](../security.md) and [Performance tracing](../tracing.md) for shared expectations. [Tsuchi](../tsuchi/overview.md) covers storage management, and [Hikari](../hikari/overview.md) covers direct inference.
