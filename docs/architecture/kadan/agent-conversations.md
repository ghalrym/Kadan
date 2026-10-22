# Agent conversations

An agent conversation separates durable work, model inference and visible message delivery. This lets the browser disconnect and the runner restart without making either the browser connection or a model response the authority for what the user has received.

| Component | Responsibility |
| --- | --- |
| Kadan API | Accept messages and expose saved conversation and work state. |
| Kadan runner | Own an assigned agent's work, prepare its context, obtain model decisions and execute enabled tools. |
| Tsuchi | Store conversations, queued runs, instruction snapshots, commands, tool results and delivery receipts. |
| Hikari | Execute inference through its existing workers and resource management. It does not execute Kadan tools or store the agent conversation. |

## Conversation selection and context

Kadan opens the oldest active conversation in the agent's Tsuchi namespace. Additional stored conversations do not create a thread picker. Agent keys match `[a-z0-9][a-z0-9_-]{0,63}` and are permanent namespace identifiers; creating an agent stores no model state.

Kadan proxies `GET /api/agents/{key}` and `PUT /api/agents/{key}` to Tsuchi for name reconciliation and renaming. The write accepts a normalized, nonblank `name` (at most 500 characters) and required `expected_name`, compared exactly with the current name inside one write transaction. A mismatch returns 409 unless the requested name is already saved, making a repeated successful request harmless. Only the name column changes; namespace keys and associated state remain intact. The browser retains the draft on failure and requires a reload before retrying. Successful saves update the roster name without changing selection or reloading the conversation.

The runner includes at most 97 recent user and assistant messages, subject to the selected model's context budget and bound tool definitions. Older history remains in Tsuchi. This separates durable records from the bounded input to one run.

Browser selection is local, and per-agent drafts use the tab's session storage. The outbox retains the original message identity, mode and persona for reconciliation and retry. Browser state is not a durable work queue.

See [Agent conversations](../../user-expectations/kadan/agent-conversations.md) for the product behavior and [Persona instructions](personas.md) for instruction snapshots.

## Model decisions and delivery

The runner uses a bounded LangGraph loop around context preparation, inference and tool execution. Hikari supplies text completions; Kadan's adapter expresses tool definitions in the model's text format and parses the result into a structured command. Kadan validates the tool and its arguments before persisting and executing it. Keeping that boundary in Kadan gives the same tool behavior to every conversation entry point without adding agent orchestration to the inference service.

Only the `send_message` tool delivers an agent message to the conversation. Completing inference, returning a tool result or finishing a run does not itself deliver a reply. This distinction prevents internal reasoning and command output from becoming user messages, and lets a run end silently when it has nothing further to communicate.

## Durable ownership and effects

Tsuchi saves a user's message and its queued run atomically. One run at a time owns a conversation; leases and version checks fence stale owners. Commands are saved before execution, and stable action IDs and delivery receipts let recovery distinguish work already delivered from work still pending. Run progress belongs in Tsuchi rather than only in process memory, so a restarted runner can resume saved work while preserving the conversation.

These receipts do not make every external effect repeatable. The [agent computer](agent-computer.md) has a separate replay constraint because its sandbox API has no idempotency keys. Changes to tool recovery must preserve that distinction.

Image generation is a separate durable job. Submission ends its initiating run instead of holding an agent turn open while Hikari works. A terminal image event queues another run in the same conversation, allowing ordinary messages to continue and the agent to deliver the eventual result. Idle conversations do not require model polling.

Hikari's scheduling, model lifecycle and memory constraints are described in [Inference architecture](../hikari/inference.md); conversation ownership must remain independent of those execution policies.

## Shared instruction file

The human-maintained `services/kadan/AGENT.md` supplies base instructions. Context preparation reads it once per new run and saves the resulting snapshot in Tsuchi; resumed runs keep that snapshot. This prevents an edit from changing the meaning of already checkpointed work.

The image includes `/app/AGENT.md`. Operators can bind-mount an edited file read-only there, or set `KADAN_AGENT_INSTRUCTION_FILE` to another readable file. Supply the same intended file to every runner. It is separate from memories and persona records, and changing it does not alter tool schemas, run limits, ownership or cancellation rules.

See [Shared agent instructions](../../user-expectations/kadan/agent-instructions.md) for user expectations.
