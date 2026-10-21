# Agent conversations

An agent conversation separates durable work, model inference and visible message delivery. This lets the browser disconnect and the runner restart without making either the browser connection or a model response the authority for what the user has received.

| Component | Responsibility |
| --- | --- |
| Kadan API | Accept messages and expose saved conversation and work state. |
| Kadan runner | Own an assigned agent's work, prepare its context, obtain model decisions and execute enabled tools. |
| Tsuchi | Store conversations, queued runs, instruction snapshots, commands, tool results and delivery receipts. |
| Hikari | Execute inference through its existing workers and resource management. It does not execute Kadan tools or store the agent conversation. |

## Model decisions and delivery

The runner uses a bounded LangGraph loop around context preparation, inference and tool execution. Hikari supplies text completions; Kadan's adapter expresses tool definitions in the model's text format and parses the result into a structured command. Kadan validates the tool and its arguments before persisting and executing it. Keeping that boundary in Kadan gives the same tool behavior to every conversation entry point without adding agent orchestration to the inference service.

Only the `send_message` tool delivers an agent message to the conversation. Completing inference, returning a tool result or finishing a run does not itself deliver a reply. This distinction prevents internal reasoning and command output from becoming user messages, and lets a run end silently when it has nothing further to communicate.

## Durable ownership and effects

Tsuchi saves a user's message and its queued run atomically. One run at a time owns a conversation; leases and version checks fence stale owners. Commands are saved before execution, and stable action IDs and delivery receipts let recovery distinguish work already delivered from work still pending. Run progress belongs in Tsuchi rather than only in process memory, so a restarted runner can resume saved work while preserving the conversation.

These receipts do not make every external effect repeatable. The [agent computer](agent-computer.md) has a separate replay constraint because its sandbox API has no idempotency keys. Changes to tool recovery must preserve that distinction.

Image generation is a separate durable job. Submission ends its initiating run instead of holding an agent turn open while Hikari works. A terminal image event queues another run in the same conversation, allowing ordinary messages to continue and the agent to deliver the eventual result. Idle conversations do not require model polling.

Hikari's scheduling, model lifecycle and memory constraints are described in [Inference architecture](inference.md); conversation ownership must remain independent of those execution policies.
