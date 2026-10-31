# Chat playground

Hikari's Chat page lets users try text inference directly. Its conversation and draft exist only in the open page and disappear on navigation or reload. It does not run agent tools or save a Kadan conversation.

Enter sends a message; Shift+Enter inserts a newline. One reply runs at a time using released BF16 LFM3-1.7B. Review its license and download it in Hikari Settings before chatting. An unavailable worker or incomplete download produces an error without choosing another model. Retry and New chat keep the same model. Loading can delay replies.

The page shows text as it arrives. Conversation capacity depends on the model's context setting, not the displayed character count. Requests that do not fit must fail rather than silently drop old messages. A reply can also stop at its output limit.

**Cancel** discards the partial reply while retaining the submitted conversation. Cleanup can finish later. **Retry last message** submits the same user turn again and can produce different text. A failed or cancelled turn must be retried or cleared before sending another message.

**New chat** clears messages and the draft. Leaving the page stops its browser request. An incomplete response must appear as an error. See [Chat API architecture](../../architecture/hikari/chat-api.md) for compatibility and [Inference operations](inference.md#conversation-prefix-reuse) for prefix reuse.
