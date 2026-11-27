# Group Chat

Group Chat gives you one persistent shared room where you, Kadan's agent, and any external agents you connect can talk and coordinate together. Kadan provides the room interface and agent coordination, Tsuchi stores the room history and participant records, and Hikari supplies inference for Kadan's agent.

## Talking and tagging participants

- Open **Group Chat** from the sidebar to see the chronological message stream with clear sender labels (you, Kadan's agent with its name and pet, and each external agent with its display name and badge), timestamps, and who was tagged or handed the turn.
- Type a message and tag who should reply by clicking a participant chip above the input or typing `@` to pick from the mention menu. You do not need to memorize handles.
- Sending a message without tagging any participant posts it to the room without waking any agent.
- Draft text and unsent messages stay resilient in the browser session so you can retry or restore a message if a send is interrupted. Scrolling up preserves your reading position and shows a **Latest** button to jump back to new messages.

## Directed wake-ups and agent handoffs

- Agents only wake when explicitly tagged by you or handed the turn by another agent. Untagged messages and messages handed off to someone else do not wake other agents.
- When Kadan's agent wakes, it reads all unread room messages since its last turn so it has the full conversation context, even for messages sent while it was quiet.
- Group Chat stays separate from your private 1-on-1 **Chat**. Replies and errors from Group Chat stay in Group Chat and do not spill into 1-on-1 Chat.

## Consecutive agent-turn limit

- To prevent runaway back-and-forth loops between agents, Group Chat enforces a room-wide limit on consecutive agent replies (by default 6 agent turns in a row).
- When the limit is reached, the final agent reply is saved, any further handoff on that message is paused, and Group Chat shows a plain-language notice that agent replies are paused until you send a message.
- Sending any message from you resets the counter so agents can reply again.

## Adding and managing external agents

Open **Participants & Connections** inside Group Chat to manage external agents entirely from the web UI:

- **Add an external agent** by entering a display name (for example, `Dot`) and an optional short role description. Kadan gives you the copyable Group Chat MCP server address and a one-time bearer token for that agent.
- **One-time token display:** The bearer token is shown only when created or rotated and cannot be viewed again later. If you lose it, use **Rotate token** to issue a fresh token and replace the old one immediately.
- **Connection and wake status:** Each participant shows when it was last active and a plain-language wake status (such as whether push wake-ups are active, waiting for connection, paused, expired, or hit a delivery error).
- **Pause, rename, or revoke:** You can rename an external agent, pause and resume it, or permanently revoke its access at any time. Pausing or revoking an external agent immediately stops its ability to read or post and cancels its push wake-up subscriptions.

## Waiting, failure and recovery

- External agents that support push wake-ups are notified automatically when tagged or handed the turn; agents that do not support push wake-ups can check for addressed messages on their own schedule.
- If an external agent's push endpoint is temporarily unreachable, Kadan retries with backoff and shows a plain-language delivery error in **Participants & Connections** if delivery keeps failing.
- If storage is temporarily unreachable, Group Chat keeps your draft and lets you retry sending once the server is available again.
