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

Before converting the reported context window into a character budget, the runner reserves `REASONING_TOKENS + REPLY_TOKENS + 1024` tokens for generation and template overhead. Input sizing uses a three-characters-per-token estimate, not exact tokenization; Hikari remains responsible for the actual token limit.

Bound tool definitions consume the same model context budget as instructions and history. Additional tool definitions can make a small configured window too narrow even for a short task; context preparation then fails visibly before inference rather than silently omitting tools or instructions. Larger windows still trim older delivered turns first.

The runner can search literal text and read older delivered user/assistant messages
through conversation tools scoped to its owned run. Results follow the existing
tool-result transcript path, retaining source roles, identities and timestamps as
evidence. Each page bounds eligible messages scanned and returned content; message
anchors require the source version, so changes reject continuation rather than
silently mixing snapshots. Coverage and content cuts are explicit. The existing
resource read fetches the full stored snapshot before these bounds are applied;
there is no new index, endpoint or independent history store.

Browser selection is local, and per-agent drafts use the tab's session storage. The outbox retains the original message identity, mode and persona for reconciliation and retry. Browser state is not a durable work queue.

The user expectations for [agent conversations](../../user-expectations/kadan/agent-conversations.md) describe the product behavior. See [Persona instructions](personas.md) for instruction snapshots.

## Request references

See [References for one response](../../user-expectations/kadan/request-references.md) for user-visible guarantees.

Chat's explicit reference picker lists active Markdown spaces and memories from the selected agent. It uses the existing document and memory list endpoints, which transport full resource records; no separate background retrieval is added. Search results are limited independently per source. Rows clear while loading or after failure, and late responses cannot replace newer query results. A run submission carries up to three `references` entries containing `resource_id` and `expected_version`, separately from message text and persona tags. Tsuchi validates their namespace, kind, active state, versions and combined 12,000-character Markdown limit in the same transaction that appends the message and queues the run. Rejection saves neither message nor run.

The durable run stores each selected resource's identity, kind, title, version and Markdown. Retrying a message identity compares the original selections before returning the saved run; it does not re-resolve edited or trashed references. Resumed runs and image completion runs retain these snapshots. Runs without selections keep their existing stored shape by omitting empty reference metadata. The browser outbox keeps the selected versions on uncertain sends and retries, clearing the selection only after confirming that submission. Saving a note does not attach references. Existing outbox entries from before reference selection are recovered with an empty selection.

The runner adds a static selection hint and offers `read_request_references`. Saved Markdown is returned as a tool result with reference-data provenance and an explicit statement that it grants no authorization and overrides no instructions. It is not concatenated into the user's message or system instructions. The established tool-response rendering remains unchanged. The model decides whether to read and use the references; selection does not guarantee use. Reading spends one command and its returned data consumes the ordinary context budget, so a request can still reach the existing command or context limit.

`bound_tools(run)` omits only `read_request_references` when the run has no selections, consistently for context preparation, transcript budgeting, inference binding and fresh-command validation. Computer availability still scopes `use_computer` and the Codex tools; sandbox tools belong to the [computer agent](computer-tasks.md). The global registry retains the reader so checkpointed calls can replay through existing ownership and cancellation fencing. No other useful tools are pruned to fit context.

## Model decisions and delivery

The runner uses a bounded LangGraph loop around context preparation, inference and tool execution. Hikari supplies text completions; Kadan's adapter expresses tool definitions in the model's text format and parses the result into a structured command. Kadan validates the tool and its arguments before persisting and executing it. Keeping that boundary in Kadan gives the same tool behavior to every conversation entry point without adding agent orchestration to the inference service.

An enabled tool's schema-validation failure is saved as a user-role rejection diagnostic, never as a pending assistant command. The next model decision receives the tool name and at most five field paths/error codes, with no argument values or validator messages. Paths retain at most four segments; unusual or oversized field names are redacted. This preserves useful evidence without retaining the rejected payload. Two correction attempts are shared across the run, including after successful calls or restart; the third rejection is saved before terminal failure. Rejections consume the command and context budgets. Ownership fencing applies to every rejection checkpoint and subsequent decision. Unknown tools, malformed command envelopes and execution failures do not enter this correction path. Saved executable actions and delivery receipts retain their existing recovery rules. See [Agent actions](../../user-expectations/kadan/agent-actions.md).

Only the `send_message` tool delivers an agent message to the conversation. Completing inference, returning a tool result or finishing a run does not itself deliver a reply. This distinction prevents internal reasoning and command output from becoming user messages, and lets a run end silently when it has nothing further to communicate.

For the LFM3 text model selected by default in the normal Compose deployment, a completed nonempty plain answer becomes a `send_message` command with `end_run=true`, only when that tool is bound. The existing run owner validates, records and delivers it through the ordinary action path. Empty output and an explicit `finish` call remain silent; malformed tool markup fails instead of being published as an answer. Native model behavior is unchanged.

## Conversation work snapshots

The Kadan conversation-runs polling response projects validated Tsuchi work records
into `AgentRunSummary` and `ImageJobSummary`. Run summaries contain identities,
trigger message/job IDs, delivery receipts, status, error and creation/update times.
Image summaries contain identities, prompt, status, error, delivery message ID and
creation/update times. The conversation resource remains part of the snapshot.

This narrows the polling response contract: internal transcripts, instructions,
tool results, leases, ownership and image-upload coordination are omitted. Clients
expecting these internal fields must update to the summary contract. Run
creation and cancellation responses, runner access and Tsuchi records remain
unchanged. Reads require an active conversation; records from another agent or
conversation fail the snapshot rather than appearing as activity. Upstream failures
remain errors instead of empty work lists. The snapshot records run state and chat
delivery; it does not establish that external actions succeeded.

## Durable ownership and effects

Tsuchi saves a user's message and its queued run atomically. One run at a time owns a conversation; leases and version checks fence stale owners. Commands are saved before execution, and stable action IDs and delivery receipts let recovery distinguish work already delivered from work still pending. Run progress belongs in Tsuchi rather than only in process memory, so a restarted runner can resume saved work while preserving the conversation.

These receipts do not make every external effect repeatable. The [agent computer](agent-computer.md) has a separate replay constraint because its sandbox API has no idempotency keys. Changes to tool recovery must preserve that distinction.

Image generation is a separate durable job. Submission ends its initiating run instead of holding an agent turn open while Hikari works. A terminal image event queues another run in the same conversation, allowing ordinary messages to continue and the agent to deliver the eventual result. Idle conversations do not require model polling.

Before uploading generated artwork, Kadan persists the job's pending upload outcome. Recovery fences the original Hikari request without resubmitting generation, then asks Tsuchi to resolve a pending upload. An unresolved outcome keeps the job active for later recovery, including cancellation. Lease expiry permits a new Kadan owner; it does not prove storage has stopped writing. After storage confirms success, Kadan verifies the deterministic asset's identity, PNG bytes and checksum before completing the job. Cancellation suppresses completion and delivery while retaining any saved artwork. Terminal events and image delivery continue to use the existing transactional event and delivery receipts. See [Image upload settlement](../tsuchi/image-uploads.md).

Hikari's scheduling, model lifecycle and memory constraints are described in [Inference architecture](../hikari/inference.md); conversation ownership must remain independent of those execution policies.

## Shared instruction file

The human-maintained `services/kadan/AGENT.md` supplies base instructions. Context preparation reads it once per new run and saves the resulting snapshot in Tsuchi; resumed runs keep that snapshot. This prevents an edit from changing the meaning of already checkpointed work.

The image includes `/app/AGENT.md`. Operators can bind-mount an edited file read-only there, or set `KADAN_AGENT_INSTRUCTION_FILE` to another readable file. Supply the same intended file to every runner. It is separate from memories and persona records, and changing it does not alter tool schemas, run limits, ownership or cancellation rules.

See the user expectations for [shared agent instructions](../../user-expectations/kadan/agent-instructions.md).
