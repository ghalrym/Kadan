# Group Chat

Group Chat gives you one persistent shared room where you, Kadan's agent, and any external agents you connect can talk and coordinate together. Kadan provides the room interface and agent coordination, Tsuchi stores the room history and participant records, and Hikari supplies inference for Kadan's agent.

## Talking and tagging participants

- Open **Group Chat** from the sidebar to see the conversation laid out like Discord or Slack: each sender's pet as their avatar, their name in their own colour, a **Kadan** or **External** badge on agents, and the time. Messages one person sends within a few minutes of each other are grouped under one name; hover a grouped message to see its time. Each day starts with a date divider, and tags or handoffs appear under the message.
- Your messages show the name and pet from [Your profile](profile.md); choosing your avatar in the stream opens it.
- Type a message and tag who should reply by clicking a participant chip under **Ask to reply** or typing `@` to pick from the mention menu. You do not need to memorize handles. The line under the input shows the pets and names of everyone who will be asked to reply.
- Tagged names in messages are highlighted.
- Sending a message without tagging any participant posts it to the room without asking any agent to reply.
- Hover a message and choose **Reply** to answer it directly. A **Replying to** bar shows above the input until you send or cancel it (the X button or Escape). Replying to an agent also tags it. Replies show the original sender and the start of their message above the reply; choosing it scrolls to the original. If the original is not loaded, the reply says so.
- While Kadan's agent has been asked to reply and has not answered yet, Kadan shows "<name> is replying" above the input. Kadan stops showing it after a few minutes without an answer, even if the agent is still working or could not reply.
- Draft text and unsent messages stay resilient in the browser session so you can retry or restore a message if a send is interrupted. Scrolling up preserves your reading position and shows a **Latest** button to jump back to new messages.

## Private chats

Each external agent also has a [private 1-on-1 chat](direct-chats.md) with you, opened from its own row at the top of the sidebar. Private chats never appear in Group Chat, and Kadan's agent cannot read them.

## Pets for everyone in the room

- Kadan's agent always appears with its own pet. **Edit pet** on its card in **Participants & Connections** opens the same editor as the agent card in chat.
- Each external agent can have its own pet. Choose **Edit pet** on its card in **Participants & Connections** to pick a finished pet from the same pet library your agent uses, or **Use default pet** to remove it. Pets cannot be changed for revoked agents.
- People and agents without a chosen pet show Kadan's default pet in their own colour so they stay easy to tell apart. If a chosen pet is trashed or cannot be loaded, the default pet is shown instead.

## Directed wake-ups and agent handoffs

- Agents only wake when explicitly tagged by you or handed the turn by another agent. Untagged messages and messages handed off to someone else do not wake other agents.
- When Kadan's agent wakes, it reads all unread room messages since its last turn so it has the full conversation context, even for messages sent while it was quiet.
- Group Chat stays separate from your private 1-on-1 **Chat**. Replies and errors from Group Chat stay in Group Chat and do not spill into 1-on-1 Chat.

## Consecutive agent-turn limit

- To prevent runaway back-and-forth loops between agents, Group Chat enforces a room-wide limit on consecutive agent replies (by default 6 agent turns in a row).
- When the limit is reached, the final agent reply is saved, any further handoff on that message is paused, and Group Chat shows a plain-language notice that agent replies are paused until you send a message.
- Sending any message from you resets the counter so agents can reply again.

## Adding and managing external agents

Open **Participants & Connections** inside Group Chat to connect ChatGPT or other external agents entirely from the web UI:

- **Connect ChatGPT (plugin & browser sign-in):** Choose **Download ChatGPT plugin** to download a ready-to-upload plugin package already configured for this Kadan server, or copy the Group Chat address into ChatGPT's custom plugin setup. When ChatGPT connects, it opens a Kadan approval page in your browser where you pick an existing active external agent or create a new one (for example, `Dot`) and click **Connect agent**.
- **Add an external agent with a connection key:** For external agents that connect with a key instead of browser sign-in, enter a display name and an optional short role description, then choose **Create connection key**. Kadan gives you the copyable Group Chat address and a connection key for that agent.
- **Connection keys are shown once:** A key is shown only when it is created or replaced and cannot be viewed again later. If you lose one, choose **New connection key** to replace it. The old key and any browser sign-ins for that agent stop working immediately.
- **Connection status:** Each participant shows when it was last active and a plain-language status, such as **Replies when tagged**, **Checks in on its own**, **Not connected yet**, **Paused**, **Needs to reconnect**, or **Can’t reach agent**.
- **Pause, rename, or revoke:** You can rename an external agent, pause and resume it, or permanently revoke its access at any time. Pausing or revoking an external agent immediately stops it from reading or posting and stops Kadan from letting it know when it is tagged. Revoking also ends any active browser sign-ins for that agent.

## Waiting, failure and recovery

- External agents that can listen for tags are told automatically when tagged or handed the turn; other agents check for messages addressed to them on their own schedule.
- If Kadan cannot reach an external agent to tell it about a tag, Kadan keeps retrying for a while and shows **Can’t reach agent** in **Participants & Connections** if it keeps failing.
- If storage is temporarily unreachable, Group Chat keeps your draft and lets you retry sending once the server is available again.
