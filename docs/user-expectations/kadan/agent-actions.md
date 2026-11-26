# Agent actions

During text conversations, Kadan agents can send messages, generate images, create Space documents, find stored files, search older chat history and manage their memories and persona. Hikari supplies the agent's decisions with the active language model; Kadan carries out the actions. Voice calls do not provide these tools.

Agents have no computer of their own. They cannot browse websites, run commands or code, or change files outside Space. For coding work, the agent can delegate threads to a connected [T3 Code plugin](plugins.md) and call tools from other connected plugins.

Kadan can look up the recorded status of this conversation's image jobs, including completed, failed and cancelled work. A chat with the same agent does not receive another conversation's task instructions. A status lookup can fail, and an earlier chat reply may describe an outdated state.

The standard deployment uses Hikari's native Inkling 2 Mini language worker (or the selected Claude/Codex Language subscription) with the same saved session, personas, tool permissions and action recovery. When using Local, its checkpoint requires a user-approved download; unavailable workers or incomplete checkpoints produce an error. Its bounded context can reject long conversations; tool-use quality and speed need manual evaluation for the work being attempted.

An action can fail because Tsuchi, Hikari, a plugin or the requested item is unavailable. Kadan returns action errors to the agent so it can decide what to do next. This does not guarantee that it will recover or send a visible explanation. Kadan rejects invalid tool arguments before executing the action and gives the model field-level feedback with up to two correction attempts per response. Repeated invalid arguments have the agent reply that it struggled to complete the task; rejected argument values are omitted. Unknown tools, malformed commands and multi-action steps are retried up to three times before the agent responds that it struggled to complete the task, retaining context of what happened only if the user asks for details. Corrections consume the normal command and context limits and do not reset after successful actions or a runner restart.

Action results have a limited text size. An interrupted or timed-out action may already have happened. After a runner restart, Kadan does not automatically replay an unconfirmed action that could change saved state; the agent is told to check before retrying.

Conversations from before the computer was removed keep their history. Earlier computer work described in them cannot be continued or inspected.

See [Stopping chat work](stopping-chat-work.md) and [operator responsibility](../security.md).
