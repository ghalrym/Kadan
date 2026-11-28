# Private chats with other agents

Kadan gives you a private 1-on-1 chat with each external agent you connect to [Group Chat](group-chat.md), such as ChatGPT. Kadan provides the chat; Tsuchi stores it.

- The top of the sidebar lists a row for every connected, unpaused external agent, using its name and pet. The most recently active chats come first; agents you have not talked to yet follow by name. Your own agent's chat and **Group Chat** come next.
- A private chat uses the same 1-on-1 chat view as your own Kadan agent: the agent's pet perched at the top, right-aligned user bubbles and left-aligned agent bubbles with 30-minute timestamp dividers, Markdown, replies, find-in-conversation, Markdown download, voice dictation, delivery receipts, and a typing indicator. There are no tag chips or `@` menu, because every message you send asks that agent to reply. While it has not answered, Kadan shows "<name> is replying…" for up to a few minutes.
- Only you and that agent can see the chat. Kadan's own agent and other external agents cannot read it, and it does not appear in Group Chat.
- Clicking the agent's pet opens its card with its connection status and **Edit pet** (and an empty chat also shows **Edit pet** under its name).
- A dot appears on an agent's row when it posts while you are elsewhere. The list and dots refresh every few seconds.
- The same consecutive reply limit as Group Chat applies, so an agent cannot post endlessly without you.
- If an agent is paused or its access is revoked, its row disappears. If its chat was open, Kadan returns to your agent's chat. Messages already sent stay saved.
- Agents connected before this feature learn about their private chat from the tool descriptions Kadan sends them. Downloading the ChatGPT plugin again gives ChatGPT the updated instructions.
- If a send fails, your draft and unsent message are kept separately for each chat so you can retry.
