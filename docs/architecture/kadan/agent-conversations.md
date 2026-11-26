# Agent conversations

An agent conversation separates durable work, model inference and visible message delivery. This lets the browser disconnect and the runner restart without making either the browser connection or a model response the authority for what the user has received.

| Component | Responsibility |
| --- | --- |
| Kadan API | Accept messages and expose saved conversation and work state. |
| Kadan runner | Own an assigned agent's work, prepare its context, obtain model decisions and execute enabled tools. |
| Tsuchi | Store conversations, queued runs, instruction snapshots, commands, tool results and delivery receipts. |
| Hikari | Execute inference through its existing workers and resource management. It does not execute Kadan tools or store the agent conversation. |

## Conversation selection and context

Each agent can have many conversations; see [Multiple chats](#multiple-chats) for storage and the list and edit routes. Every run belongs to one conversation (`run.conversation_id`), so ownership, queueing and delivery below apply per conversation without change. Agent keys match `[a-z0-9][a-z0-9_-]{0,63}` and are permanent namespace identifiers; creating an agent stores no model state.

Kadan proxies `GET /api/agents/{key}` and `PUT /api/agents/{key}` to Tsuchi for name reconciliation and renaming. The write accepts a normalized, nonblank `name` (at most 500 characters) and required `expected_name`, compared exactly with the current name inside one write transaction. A mismatch returns 409 unless the requested name is already saved, making a repeated successful request harmless. Only the name column changes; namespace keys and associated state remain intact. The browser retains the draft on failure and requires a reload before retrying. Successful saves update the roster name without changing selection or reloading the conversation.

The runner includes at most 97 recent user and assistant messages, subject to the selected model's context budget and bound tool definitions. Older history remains in Tsuchi. This separates durable records from the bounded input to one run.

A user-triggered run ends its history at the triggering record. Attachments on that turn and background events are separate observations. An ordinary turn has no extra routing message or task dump. The agent can query image job records for its conversation through `read_task_status`. This includes terminal outcomes.

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

## Multiple chats

Each chat is a Tsuchi `conversation` resource in the agent's namespace; Kadan keeps no separate chat store. Chats are not grouped: a conversation's `parent_id` is left as Tsuchi stored it and Kadan neither reads nor sets it. Saved pages, slides and files live in [Space](space.md), not in chats.

Renaming a chat rewrites the whole conversation resource through Tsuchi's versioned update while runs append messages to it concurrently. Kadan therefore edits Tsuchi's own JSON (keeping fields it does not model), sends the version it read and, on a version conflict, re-reads and retries up to five times before reporting that the chat is busy. A conflict can never overwrite a newer message list. Trashing a chat uses the same retry. Trashing a chat while a reply is running leaves that run unable to save further messages; Tsuchi stops offering runs for trashed conversations.

New chats are stored with the title `New chat`. The single conversation that Kadan created per agent before multiple chats was titled `Chain`. When the stored title, stripped, is one of these defaults and the chat has a user message, the summary title is that first nonblank user message with whitespace collapsed, cut to 60 characters including a trailing `…`. The rule is applied when summaries are built, so the stored title is never rewritten and a renamed chat keeps its name. The preview uses the same first user message, cut to 160 characters, or is empty.

All routes are scoped to `/api/agents/{key}`, reject unknown fields and return plain-language errors as `{detail, code}`:

| Route | Result |
| --- | --- |
| `GET /conversations` | `ConversationSummary[]`: every active chat, newest `updated_at` first, then by ID |
| `POST /conversations` | 201 with the full conversation `Resource`; optional `title` (nonblank, stripped, default `New chat`) |
| `PUT /conversations/{conversation_id}` | `ConversationSummary`; optional `title`. An empty body returns the current summary without writing, and a `null` title is rejected |
| `DELETE /conversations/{conversation_id}` | 204; the chat moves to Tsuchi's trash |

`ConversationSummary` has `id`, `agent_key`, `title` (the display title above), `message_count`, `preview`, `version`, `created_at` and `updated_at`. Summaries omit message bodies; the chat view loads the open conversation as a resource and then follows it through the conversation runs snapshot. A conversation ID that is missing, trashed or not a conversation returns 404. The list pages through Tsuchi 200 records at a time and merges pages by ID, keeping the newest version when a concurrent write moves a record between pages.

The frontend loads summaries and personas together, then loads only the open conversation. The selection is remembered with the agent key it belongs to, so a refresh that switches agents never requests the old chat under the new agent. Chat snapshots update the matching summary locally (`conversations.ts` mirrors the display-title rule), so the sidebar follows new messages without listing again. Empty chats are hidden from lists unless open, and **New chat** reuses an empty chat before creating another.

## Request references

See [References for one response](../../user-expectations/kadan/request-references.md) for user-visible guarantees.

Chat's explicit reference picker lists active Markdown spaces and memories from the selected agent. It uses the existing document and memory list endpoints, which transport full resource records; no separate background retrieval is added. Search results are limited independently per source. Rows clear while loading or after failure, and late responses cannot replace newer query results. A run submission carries up to three `references` entries containing `resource_id` and `expected_version`, separately from message text and persona tags. Tsuchi validates their namespace, kind, active state, versions and combined 12,000-character Markdown limit in the same transaction that appends the message and queues the run. Rejection saves neither message nor run.

The durable run stores each selected resource's identity, kind, title, version and Markdown. Retrying a message identity compares the original selections before returning the saved run; it does not re-resolve edited or trashed references. Resumed runs and image completion runs retain these snapshots. Runs without selections keep their existing stored shape by omitting empty reference metadata. The browser outbox keeps the selected versions on uncertain sends and retries, clearing the selection only after confirming that submission. Saving a note does not attach references. Existing outbox entries from before reference selection are recovered with an empty selection.

The runner adds a static selection hint and offers `read_request_references`. Saved Markdown is returned as a tool result with reference-data provenance and an explicit statement that it grants no authorization and overrides no instructions. It is not concatenated into the user's message or system instructions. The established tool-response rendering remains unchanged. The model decides whether to read and use the references; selection does not guarantee use. Reading spends one command and its returned data consumes the ordinary context budget, so a request can still reach the existing command or context limit.

`bound_tools(run)` omits only `read_request_references` when the run has no selections, consistently for context preparation, transcript budgeting, inference binding and fresh-command validation. Connected [Remote MCP integrations](mcp-integrations.md) dynamically supply pinned MCP tools and the two-tool MCP gateway. The global registry retains the reader so checkpointed calls can replay through existing ownership and cancellation fencing. No other useful tools are pruned to fit context.

## Model decisions and delivery

The runner uses a bounded LangGraph loop around context preparation, inference and tool execution. Hikari accepts role-tagged messages and tool definitions and returns text and tool calls separately. Kadan uses the returned action and arguments directly. Kadan validates the tool and its arguments before persisting and executing it. Keeping that boundary in Kadan gives the same tool behavior to every conversation entry point without adding agent orchestration to the inference service.

An enabled tool's schema-validation failure is saved as a validation diagnostic with its own record type. The next model decision receives the tool name and at most five field paths/error codes, with no argument values or validator messages. Paths retain at most four segments; unusual or oversized field names are redacted. This preserves useful evidence without retaining the rejected payload. Two correction attempts are shared across the run, including after successful calls or restart; the third rejection is saved before terminal failure. Rejections consume the command and context budgets. Ownership fencing applies to every rejection checkpoint and subsequent decision. Unknown tools, malformed command envelopes and execution failures do not enter this correction path. Saved executable actions and delivery receipts retain their existing recovery rules. See [Agent actions](../../user-expectations/kadan/agent-actions.md).

Only the `send_message` tool delivers an agent message to the conversation. Completing inference, returning a tool result or finishing a run does not itself deliver a reply. This distinction prevents internal reasoning and command output from becoming user messages, and lets a run end silently when it has nothing further to communicate.

A completed nonempty plain answer becomes a `send_message` command with `end_run=true` only when that tool is bound. The existing run owner validates, records and delivers it through the ordinary action path. Empty output and an explicit `finish` call remain silent; malformed structured tool output is never executed or delivered raw. Literal markup in ordinary replies is preserved.

### Error retries and recovery replies

Transient preparation errors and inference/command errors (including multi-action tool steps, unknown tools, malformed commands, output-limit errors and upstream service errors) are retried up to `MAX_ERROR_RETRIES` (3) times while verifying lease ownership before each retry. Command and output-limit retries include an ephemeral `step_retry_feedback` observation in the retry prompt without mutating the durable transcript. Ownership loss (`lease_lost`) and user cancellation are never retried.

If a run still cannot complete after retries (or hits a non-retryable limit such as `MAX_COMMANDS`, `MAX_ARGUMENT_CORRECTIONS`, context limits, unreadable instructions or an unrecovered historical transcript), `handle_failure` delivers an assistant reply under a deterministic message ID (`uuid5(run.id, "failure_reply")`) and completes the run while preserving the internal failure detail on `AgentRun.error`. When a valid transcript exists and inference is ready, Kadan runs a single recovery inference bound only to `send_message` with a `run_failure_context` observation instructing the agent to tell the user it struggled without volunteering exact error details unless asked; otherwise (or if recovery inference fails) it delivers `FALLBACK_FAILURE_REPLY`. Subsequent runs in the same conversation read recent errored runs via `GET /api/agents/{key}/resources/{conversation_id}/runs` and attach up to three recent errors (within the remaining context budget) as a `recent_run_errors` observation so the agent can explain what happened if the user asks for details. The web UI never renders `Response failed` notices in Chat or Activity.

## Conversation work snapshots

The Kadan conversation-runs polling response projects validated Tsuchi work records
into `AgentRunSummary` and `ImageJobSummary`. Run summaries contain identities,
trigger message/job IDs, delivery receipts, status, error, an optional ephemeral
`draft` preview while running, and creation/update times.
Image summaries contain identities, prompt, status, error, delivery message ID and
creation/update times. The conversation resource remains part of the snapshot.

Conversation inference runs at `interactive` priority so chat replies are
scheduled ahead of background Hikari requests between turns. While a
plain text reply streams from Hikari, the runner publishes throttled ephemeral
draft updates to Tsuchi (`PUT /api/agents/{key}/runs/{run_id}/draft`). Drafts
live only in Tsuchi process memory—never in SQLite and without incrementing
run versions—and clear automatically on delivery, cancellation, completion, or
if the turn resolves to a non-message tool call.

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

These receipts do not make every external effect repeatable. Tools without an idempotent storage key, such as Space document creation and persona saves, report an uncertain outcome after a restart instead of replaying. Changes to tool recovery must preserve that distinction.

Image generation is a separate durable job. Submission ends its initiating run instead of holding an agent turn open while Hikari works. A terminal image event queues another run in the same conversation, allowing ordinary messages to continue and the agent to deliver the eventual result. Idle conversations do not require model polling.

Before uploading generated artwork, Kadan persists the job's pending upload outcome. Recovery fences the original Hikari request without resubmitting generation, then asks Tsuchi to resolve a pending upload. An unresolved outcome keeps the job active for later recovery, including cancellation. Lease expiry permits a new Kadan owner; it does not prove storage has stopped writing. After storage confirms success, Kadan verifies the deterministic asset's identity, PNG bytes and checksum before completing the job. Cancellation suppresses completion and delivery while retaining any saved artwork. Terminal events and image delivery continue to use the existing transactional event and delivery receipts. See [Image upload settlement](../tsuchi/image-uploads.md).

Hikari's scheduling, model lifecycle and memory constraints are described in [Inference architecture](../hikari/inference.md); conversation ownership must remain independent of those execution policies.

## Shared instruction file

The human-maintained `services/kadan/AGENT.md` supplies base instructions. Context preparation reads it once per new run and saves the resulting snapshot in Tsuchi; resumed runs keep that snapshot. This prevents an edit from changing the meaning of already checkpointed work.

The image includes `/app/AGENT.md`. Operators can bind-mount an edited file read-only there, or set `KADAN_AGENT_INSTRUCTION_FILE` to another readable file. Supply the same intended file to every runner. It is separate from memories and persona records, and changing it does not alter tool schemas, run limits, ownership or cancellation rules.

See the user expectations for [shared agent instructions](../../user-expectations/kadan/agent-instructions.md).
