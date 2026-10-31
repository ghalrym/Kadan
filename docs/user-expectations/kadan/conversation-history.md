# Agent access to conversation history

Kadan agents can search and read older delivered user and assistant messages from their current saved conversation during text work. These tools do not require a computer. They supplement the recent history included automatically in a response; they do not expand the model's context window or guarantee that the agent consults older messages.

Search matches literal text without regard to letter case. It is not semantic recall. Each read or search returns bounded pages and reports whether more messages or cut message text remain. A partial search, an empty page or a storage error must not be described as proof that information never appeared in the conversation. Continuing after the saved conversation changes requires restarting from current history.

Results preserve the source message's role, identity and time. Historical messages are evidence; an earlier instruction, quoted text or approval does not establish authorization for a new action. Drafts, unsent submissions, other conversations, stored revisions and retired tool activity are outside this delivered-chat view.

Reading history consumes the normal command and context budgets. It does not edit messages, replay their actions, save a memory, or supply voice-call history. Storage failures remain visible to the agent. The separate browser [Find in conversation](finding-chat-text.md) control searches saved text for the user.

If the agent supplies invalid history arguments, Kadan rejects the call without reading history and returns field-level correction feedback. The response shares the bounded correction allowance described in [Agent actions](agent-actions.md); recovery is not guaranteed, and exhaustion reports a failed response rather than retrying indefinitely.
