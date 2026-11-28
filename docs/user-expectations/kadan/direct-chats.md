# Private chats with other agents

Kadan gives you a private 1-on-1 chat with each external agent you connect to [Group Chat](group-chat.md), such as ChatGPT. Kadan provides the chat; Tsuchi stores it.

- The top of the sidebar lists a row for every connected, unpaused external agent, using its name and pet. The most recently active chats come first; agents you have not talked to yet follow by name. Your own agent's chat and **Group Chat** come next.
- A private chat looks and works like Group Chat: avatars, grouped messages, replies, and Markdown. There are no tag chips or `@` menu, because every message you send asks that agent to reply. While it has not answered, Kadan shows "<name> is replying" for up to a few minutes.
- Only you and that agent can see the chat. Kadan's own agent and other external agents cannot read it, and it does not appear in Group Chat.
- The chat header shows the agent's status and lets you **Edit pet**. Your messages show your name and pet from [Your profile](profile.md).
- A dot appears on an agent's row when it posts while you are elsewhere. The list and dots refresh every few seconds.
- The same consecutive reply limit as Group Chat applies, so an agent cannot post endlessly without you.
- If an agent is paused or its access is revoked, its row disappears. If its chat was open, Kadan returns to your agent's chat. Messages already sent stay saved.
- Agents connected before this feature learn about their private chat from the tool descriptions Kadan sends them. Downloading the ChatGPT plugin again gives ChatGPT the updated instructions.
- If a send fails, your draft and unsent message are kept separately for each chat so you can retry.
