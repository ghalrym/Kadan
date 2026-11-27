# Inkling 2 Mini coding harness

See [Work threads and coding assistants](../../user-expectations/kadan/work.md) for user-visible behavior and [Native Work threads](work.md) for the shared `HarnessAdapter` / `WorkRunner` contract.

## Purpose and module layout

Unlike external CLI wrappers, Kadan's built-in **Inkling (Local)** harness (`harness_id = "inkling"`) runs directly against Hikari's resident C++ Strata MoE engine (`hikari-worker-llm`) serving Inkling 2 Mini. It is organized into four focused modules under `services/kadan/kadan/work/`:

- `inkling_adapter.py`: `InklingHarnessAdapter` and `InklingHarnessSession` — availability discovery against Hikari `/v1/models`, streaming agent loop, live steering, cancellation, session state persistence/hydration, and telemetry surfacing.
- `inkling_tools.py`: Compact 8-tool coding surface, workspace containment, bounded head/tail output capture, exact-match edit mismatch diagnostics, and Pydantic guided argument-validation recovery.
- `inkling_budget.py`: Conservative token estimation against `HIKARI_LLM_CONTEXT_LIMIT`, atomic tool-call group partitioning, and structured working-memory compaction.
- `inkling_decisions.py`: Single-pass `POST /v1/decisions` guardrails for command approval gating, loop/stuck detection, and completion verification.

## Base-weights routing and Hikari scheduling harmony

- Every `/v1/chat/completions` (turn step and compaction summary) and `/v1/decisions` request sets `"weight_variant": "base"` resolved through `resolve_task_weight_variant(task_class="harness")` (#528). Direct persona chat continues using abliterated weights while Hikari hot-swaps only the 34 differing tensors in place without reloading `hikari-worker-llm`.
- `WorkRunner` sets `SessionConfig.priority = "background"` whenever a Work thread was launched autonomously by a chat agent (`thread.origin_agent_key is not None`) and `"interactive"` when a user drives the thread directly. Under Hikari's fairness-bounded scheduler (`docs/architecture/hikari/worker-llm.md`), interactive chat stays responsive without mid-generation preemption.
- Availability checks inspect Hikari `GET /v1/models` specifically for the same-precision **base** Inkling 2 Mini checkpoint (`inkling-2-mini-iq4-nl` or `inkling-2-mini-q8-0`) rather than assuming an abliterated download satisfies coding runs.
- Context-limit, VRAM/RAM admission, and missing-checkpoint errors returned by Hikari are translated into plain-language `HarnessError` messages without silent truncation or model substitution.

## Prefix-cache and GPU boundary snapshot invariants

`hikari-worker-llm` keeps a bounded host compacted checkpoint pool and two GPU-resident boundary snapshots (~164 MiB each) that enable zero-copy KV rewind (`checkpoint_hit` and `static_prefix_hit`). To maximize cache hits across steps and across turns:

1. **Byte-identical static prefix:** `STATIC_SYSTEM_PROMPT` and `INKLING_TOOL_SCHEMAS` (sorted canonical JSON keys, immutable tuple) contain zero volatile data (no timestamps, working directories, git state, or shifting tool lists). Before every inference call, `_assert_prefix_invariant()` verifies the SHA-256 digest of `(messages[0]["content"], canonical_tool_schemas_json())` matches `STATIC_PREFIX_DIGEST`.
2. **Append-only turn history:** User prompts, live steering updates, assistant tool calls, and tool results are appended strictly in order until explicit compaction.
3. **Stable conversation key:** Every step passes `conversation_id = f"inkling:{thread_id}"` and `reuse_prefix = True`. `InklingHarnessAdapter` retains exact in-process `InklingSessionState` per `thread_id` across turns for exact `checkpoint_hit` reuse, and hydrates prior user/assistant exchanges from Tsuchi activity onto the same immutable static prefix after a runner restart (`static_prefix_hit`).
4. **Telemetry surfacing:** Each step emits a `status` activity event carrying `raw["telemetry"]` (`weight_variant`, `priority`, `static_prefix_sha256`, `cache_hit`, `cache_reason`, `reused_tokens`, `stored_tokens`, `weight_swap_ms`, `prefill_ms`, `cache_restore_ms`, `image_encode_ms`, `decode_tokens_per_second`, `ttft_ms`, `duration_ms`, `prompt_tokens`, `context_limit`).

## Compact coding tool surface

Inkling 2 Mini sees exactly eight low-ambiguity tools:

| Tool | Purpose | Key guardrails |
| --- | --- | --- |
| `read_file` | Read 1-based line ranges (`start_line`, `end_line`) | Clamped to `MAX_READ_LINES` (300) and `MAX_TOOL_RESULT_CHARS` (12,000); rejects binary files and paths outside `working_dir` |
| `find_files` | Glob search under `working_dir` | Skips `.git`, `node_modules`, `.venv`, caches; capped at `MAX_FIND_RESULTS` (100) |
| `search_text` | Text/regex search (`rg` with pure-Python fallback) | Scoped to `working_dir`, capped at `MAX_SEARCH_MATCHES` (50) |
| `edit_file` | Exact-match targeted replacement (`replace_all` optional) | Rejects ambiguous matches with line numbers; on miss returns top nearby line windows via `difflib` for one-step self-correction |
| `write_file` | Create or overwrite a file inside `working_dir` | Auto-creates parent directories; emits `file_change` (`created`/`modified`) |
| `run_command` | Workspace-scoped shell command | Explicit timeout (default 60s, max 300s), own process group killed cleanly on timeout/cancel, head/tail output capping, approval gated |
| `git_inspect` | Focused `status`, `diff` (optional `--cached`), or `log` | Bounded read-only Git inspection without arbitrary shell execution |
| `ask_user` | Pause thread with structured `pending_question` | Supports clickable options; relayed to originating chat conversation if agent-launched |

- **Output bounding (`cap_tool_output`):** Every tool result exceeding 12,000 characters is capped with 60% head + 40% tail preservation and an explicit truncation notice before entering prompt history.
- **Guided argument-validation recovery (`validate_tool_arguments`):** Unknown tools, malformed JSON, and Pydantic `ValidationError`s (`extra="forbid"`) return a structured `tool` error message listing field problems and a recovery hint instead of aborting the turn. Four consecutive validation failures abort cleanly to prevent runaway loops.

## Explicit token budgeting and structured working-memory compaction

- Before every step, `compute_budget_snapshot` estimates total prompt + tool + image tokens against the active `HIKARI_LLM_CONTEXT_LIMIT` (discovered from Hikari `/v1/models`, default `65536`, clamped to `[4096, 1048576]`) while reserving output headroom (`4096` tokens, capped to 25% of small windows).
- When utilization reaches `72%`, `_compact_history` partitions history via `partition_messages_for_compaction`, keeping `messages[0]` (the static system prompt) untouched and keeping assistant `tool_calls` + their `tool` results atomic so tool results are never orphaned.
- Compaction requests a concise working-memory summary from Hikari (`weight_variant="base"`, `reuse_prefix=False`, `reasoning_effort="low"`) with five fixed sections: **Goal**, **Modified files**, **Verified checks**, **Open blockers**, and **Next steps** (falling back to a deterministic summary built from `WorkingMemoryTracker` if the summary call fails).
- `rebuild_compacted_messages` resets history onto `[STATIC_SYSTEM_PROMPT, handoff_message, *recent_tail]` and re-verifies `STATIC_PREFIX_DIGEST`.
- If the prompt still exceeds `available_prompt_tokens` after compaction, the turn fails with a clear `HarnessError` rather than silently dropping messages.

## Native single-pass Decisions API guardrails

`inkling_decisions.py` uses Hikari's zero-decode `POST /v1/decisions` (`weight_variant="base"`), which snapshots and restores the resident chat KV prefix (`docs/architecture/hikari/worker-decision.md`):

1. **Command safety / approval gating (`classify_command_approval`):** Hard-stop destructive patterns (`rm -rf`, `git push --force`, `git reset --hard`, `sudo`, remote shell pipes, recursive `chmod`/`chown`) always emit `pending_approval` regardless of session approval. Otherwise, ambiguous commands are classified with a `Noul` `requires_approval` question (failing safe to requiring approval if Decisions is unreachable).
2. **Loop / stuck detection (`evaluate_loop_stuck`):** Triggered only when local heuristics see 3 consecutive tool errors or 3+ repeated call signatures in the last 6 steps. Evaluates `is_stuck` and `should_ask_user` in one shared-state pass to either inject a structured recovery nudge or emit `pending_question`.
3. **Verification / completion gating (`evaluate_completion_verification`):** Triggered at most once per turn when files were edited (`unverified_edits == True`) and the model attempts to finish without running a command after its last edit. Evaluates `needs_verification` and injects a concise verification reminder when needed.

## Multimodal attachments

`WorkRun.images` stores validated PNG/JPEG data URLs (up to 16 per turn). `WorkRunner` forwards `run.images` to sessions declaring `supports_images = True`, and `InklingHarnessSession` encodes them as Hikari's ordered `[{"type": "text", ...}, {"type": "image_url", ...}]` user message parts so Inkling 2 Mini's built-in `mmproj` vision encoder processes screenshots natively.
