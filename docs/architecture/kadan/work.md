# Native Work threads and coding harness wrapper

See [Work threads and coding assistants](../../user-expectations/kadan/work.md) for the user-visible behavior.

## Responsibilities and boundaries

- **Tsuchi (`services/tsuchi/`):** Owns durable storage for `work_projects`, `work_threads`, `work_runs`, `work_activity`, and `work_harnesses`, plus optimistic locking (`version`), `client_request_id` idempotency, snooze wakeup queries (`POST /api/work/threads/wake-snoozed`), and `RunEvent(kind="work")` chat wakeup runs.
- **Kadan API (`services/kadan/kadan/api/routes/work.py`, `services/kadan/kadan/work/client.py`):** Exposes `/work/*` HTTP endpoints, interactive server folder browsing (`/work/filesystem/browse`), repository cloning, Git worktree lifecycle (`WorktreeManager`), and harness availability/sign-in endpoints.
- **Kadan Runner (`services/kadan/kadan/work/runner.py`):** Runs `WorkRunner` alongside `AgentRunner` and `T3ThreadEvents`. It reconciles expired snoozes, claims queued/lease-expired `WorkRun`s, executes turns through the selected `HarnessAdapter`, persists normalized activity and pending requests, and emits idempotent Tsuchi `event-runs` (`kind="work"`) into originating chat conversations without polling.
- **Frontend (`services/frontend/src/kadan/`):** Renders the 4-bucket Work section in `Sidebar.tsx`, the live thread workspace in `WorkThreadView.tsx`, and the New Thread / Projects / Harnesses setup UI in `WorkSetup.tsx`, plus in-app `#work/<thread_id>` link navigation from chat Markdown.

## Extensible harness adapter contract

`services/kadan/kadan/work/contract.py` defines the stable contract implemented by every coding harness (including future local harnesses such as Inkling 2 Mini in #530):

- `HarnessAdapter`:
  - `check_availability(services) -> HarnessAvailability`: Reports binary installation, UI auth state, model catalog, and capabilities (`supports_steer`, `supports_resume`, `supports_approvals`, `supports_questions`).
  - `start_sign_in(services, payload)`, `sign_in_status(services)`, `submit_sign_in(services, payload)`, `sign_out(services)`: UI-driven auth lifecycle backed by Tsuchi `/api/work/harnesses/{harness}` (raw credentials are never returned by public Kadan routes).
  - `open_session(config: SessionConfig, services) -> HarnessSession`: Opens or resumes a live session bound to `config.working_dir` (project root or isolated Git worktree).
- `HarnessSession`:
  - `session_id`: Durable provider session/thread ID persisted on `WorkThread.harness_session_id` for resume across turns and restarts.
  - `start_turn(prompt) -> AsyncIterator[NormalizedEvent]`: Streams uniform events (`session_started`, `text_delta`, `reasoning_delta`, `assistant_message`, `tool_call`, `tool_result`, `file_change`, `pending_question`, `pending_approval`, `status`, `error`, `turn_completed`).
  - `steer(prompt) -> bool`: Optional inline live steering; returning `False` causes `WorkRunner` to run the follow-up turn sequentially.
  - `interrupt()`: Requests immediate cancellation of the active turn.
  - `respond_pending(request_id, response)`: Delivers user answers or permission decisions back to the harness.
  - `close()`: Cleans up child processes and temporary auth homes.
- `HarnessRegistry` (`services/kadan/kadan/work/registry.py`): Registers adapters by `harness_id`.

## Bundled wire transports

1. **Codex (`codex_adapter.py`):** Drives `codex app-server --listen stdio://` over newline-delimited JSON-RPC (`stdio_jsonrpc.py`) in full coding-harness mode (`thread/start`, `thread/resume`, `turn/start`, `turn/steer`, `turn/interrupt`, plus server-initiated approval and `requestUserInput` round-trips). Browser device-code auth (`codex login --device-auth`) is driven by `auth_flows.py`.
2. **Claude Code (`claude_adapter.py`):** Drives `claude -p --input-format stream-json --output-format stream-json --verbose --include-partial-messages --permission-prompt-tool stdio` in full coding-harness mode with `--resume`, `control_request` / `control_response` for tool approvals and `AskUserQuestion`, and `interrupt`. Browser `setup-token` auth is driven via PTY in `auth_flows.py`.
3. **Shared ACP client (`acp_client.py`):** Bidirectional JSON-RPC 2.0 stdio client that negotiates **ACP v2** first (`initialize` with `protocolVersion: 2` and `session/resume`) and automatically falls back to **ACP v1** (`protocolVersion: 1` and `session/load`) when a server rejects v2. Powers both `AntigravityHarnessAdapter` (`antigravity_adapter.py`, `antigravity-acp`) and `GrokHarnessAdapter` (`grok_adapter.py`, `grok --acp`).

## Replay safety and event-driven chat wakeups

- Native agent tools in `services/kadan/kadan/runner/tools/work.py` (`work_project_list`, `work_thread_list`, `work_thread_read`, `work_thread_launch`, `work_thread_send`, `work_thread_interrupt`, `work_pending_request`, `work_thread_organize`) pass deterministic `client_request_id` values derived from `conversation.action_id` so runner restarts replay safely without duplicating threads or turns.
- When an agent-launched thread pauses on a pending request or reaches a terminal run state, `WorkRunner` emits an idempotent `POST /api/agents/{key}/resources/{conversation_id}/event-runs` with `kind="work"` (`uuid5(run.id, ...)`). `AgentRuns.context()` injects it as `work_thread_event`.
- The external `t3_code` MCP preset (`mcp_presets.py`) and `T3ThreadEvents` watcher (`t3_events.py`) remain intact under **Plugins** so both integrations coexist cleanly.
