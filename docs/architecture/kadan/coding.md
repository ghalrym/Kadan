# Codex coding jobs

The [coding expectations](../../user-expectations/kadan/coding.md) describe the
user experience. Kadan delegates to the official pinned Codex CLI inside the
existing shared computer. Hikari still supplies conversation inference; Tsuchi's
conversation ownership and chat cancellation are unchanged.

[Codex exec JSON](https://developers.openai.com/codex/noninteractive) provides bounded noninteractive turns and explicit saved-session
resume without introducing an app-server connection, approval protocol or another
service. Jobs use workspace-write and never request interactive approvals. The
Codex model is `gpt-6.1-sol` with high reasoning effort. Its access to other project
directories and external commands remains subject to Codex's execution policy.
There is no approval relay or automatic escalation when that policy denies an action.

The sandbox image installs official npm package `@openai/codex@0.162.1` and copies
`coding_jobs.py` to `/opt/kadan`. No package or credentials are installed by a tool
call. Both setup probes and execution use `/home/gem/.codex` as `CODEX_HOME`, on
the existing home volume. Setup checks the exact CLI version and the exit status
of `codex login status`, discarding authentication diagnostics. It makes no model
request. The Coding panel starts `codex login --device-auth` in a detached
computer process, displays its device link and code, and polls local login status.
The login control stores the process identity on the computer so later requests can
check or stop the same login. Sign-out invokes `codex logout` on that computer.
Kadan does not copy credentials from its API host, browser or this development environment.

The API's `CodingClient` and the runner's three tools invoke the same supervisor
through the sandbox's existing Bash exec route. The control invocation receives
base64 encoded JSON rather than shell-interpolated task text, returns structured
JSON, and closes its disposable Bash session so status polling cannot exhaust the
computer's session capacity. The transport client reports uncertain outcomes
without retrying. Only the scoped delegation wait retries its idempotent action
or reads as described below. Raw shell stderr and auth output do not enter API errors.

`codex_status` reads setup/recent job IDs or one job's events after a cursor.
`codex_start` accepts a prompt, absolute existing working directory, deadline and
optional previous job ID. It is a conversation-scoped tool, bound only when a
computer is configured. `RunConversation` derives the supervisor request/job ID
from the persisted run ID and command position, using the same action-identity
scheme as message delivery. The model cannot choose an ID from another task.
`codex_cancel` records a durable cancellation request.

Delegation keeps its original Tsuchi run and saved command pending until the job
returns a terminal outcome. The runner waits asynchronously between status reads;
lease renewal continues independently and waiting consumes no additional model
calls or commands. A restart recovers the saved action using the supervisor's
idempotent submission contract. The completed result is checkpointed as that
action's tool result before ordinary bounded inference resumes to finish the task.
This avoids a second completion queue, event run, or cross-store delivery receipt.
Existing message action IDs still deduplicate a reply after uncertain delivery.
Later user messages wait behind this run, matching existing per-agent scheduling.

Every wait iteration checks the original run's identity, agent, conversation,
ownership and live lease before and after the computer operation. Chat cancellation
or ownership loss stops the wait and fences result checkpoints, inference and
delivery; it neither cancels Codex nor queues a replacement run. Browser selection
is never consulted for continuation. Job reads and writes validate the returned job
and request IDs against the requested action. Repeated progress or terminal reads
cannot generate another event run.

Unconfirmed transport operations retry under that same action ID without inference.
The wait is bounded by the job's original creation time plus its timeout and a
two-minute recovery allowance, with a local bound while submission is uncertain.
Persisted job creation time prevents restarts from extending that allowance.
A missing job, setup rejection or invalid response returns an error; an expired
recovery deadline returns an unconfirmed outcome, never automatic resubmission
under another ID. Only `codex_start` gains this idempotent replay path; other
computer mutations keep their existing no-replay rule.

The web API exposes `GET /api/computer/coding`,
`GET /api/computer/coding/{job_id}?after=…` and
`POST /api/computer/coding/{job_id}/cancel`. The browser cannot submit a coding
prompt through these routes. Coding controls inherit the computer's existing
deployment access model and simple API-key transport; they introduce no account system.

The supervisor owns atomic JSON records beneath
`/home/gem/.local/state/kadan/coding`, with a global file lock serializing admission
and updates across API and runner processes. A fingerprint compares repeated IDs
against the original prompt, supplied directory, deadline and previous job. The
record exists before worker launch. Prompts are temporary private handoff files,
removed when consumed or recovered; completed records keep a fingerprint rather
than the prompt. The detached worker survives control-request and Kadan restarts.
Job history is retained on disk for idempotency; the panel returns at most twenty
recent jobs. Each record retains at most one hundred summarized events; reads
return at most twenty with an explicit truncation indicator and cursor.

The worker acquires Linux subreaper ownership before launching Codex. It consumes
bounded JSONL lines, records the session ID, summarizes progress and requires both
a successful process exit and a completed turn event before reporting completion.
Deadlines, durable cancellation and supervisor signals all enter process cleanup.
Cleanup repeatedly discovers owned descendants, uses identity-checked pidfds to
signal them, escalates TERM to KILL and reaps children, including detached adopted
descendants. Only confirmed cleanup releases admission. A missing worker or failed
cleanup conservatively blocks with `cleanup_required`; the container incarnation
(kernel boot ID and PID 1 start time) establishes old-process termination after a
computer restart. Kadan does not attempt to undo edits or replay uncertain work.

Resume requires a terminal prior job with confirmed cleanup, a recorded session
ID and the same resolved directory. It invokes `codex exec resume` with that
explicit ID, never `--last`, and does not create a replacement session on failure.

The Coding panel's explicit continuation reuses the conversation run submission
route with `resume_coding_job_id`. Kadan resolves the prior job's directory rather
than accepting a browser-supplied directory. Tsuchi atomically saves the user
message and initial `codex_start` assistant command in the existing transcript in
one transaction. The enqueue input is `initial_command`; it is not a persisted
field. Retries compare that immutable first checkpoint as well as the message
identity, even after the run advances. The browser uses the conversation's existing
outbox to retain the same job, prompt and message ID after an uncertain submission.
Messages can retry that same typed request without downgrading it to ordinary chat.

For these explicit runs, the runner replays the seeded `codex_start` and uses the same
delegation wait, ownership fencing and request ID recovery as model-selected jobs.
It then validates the persisted result's action ID and renders a fixed status
report with the returned summary and identifiers. It does not call Hikari to select
the action or write the report. Saved command and delivery identities still dedupe
recovery before and after either checkpoint. Ordinary model-directed chat is
unchanged: a free-form reply can omit a requested action and is not execution
evidence. This is why continuation has a separate explicit user action rather than
phrase detection or a stronger prompt alone.

Tsuchi treats the initial command as an opaque assistant checkpoint; Kadan owns
its tool schema. A leading assistant `codex_start` with an explicit previous job
identifies a continuation. Ordinary model-directed runs start with their system
context. Neither user prose nor special identifiers select this path. Tsuchi
protects the saved prefix on the first claim as well as later updates; recovery
of an expired lease cannot change the transcript at all.

Deploy Kadan API/runner/web and Tsuchi together for this contract. Tsuchi needs
the optional enqueue input and first-claim prefix protection. Older enqueue APIs
reject the new input before creating work; ordinary submissions omit it. Stored
JSON and SQL schemas are unchanged, and old strict readers can deserialize the
seeded records without conversion. `initial_command` is a computed property, not
a serialized field.

Cutover requires quiescing submissions and runners after coordinating active
work, taking a consistent SQLite backup including WAL state, then updating Tsuchi
and Kadan API/runner/web before reopening submissions. The sandbox execution
protocol is unchanged. Reader compatibility does not make mixed runners safe:
old runners can execute the saved command, then use model-written reporting
instead of the new fixed result report, and old storage has weaker first-claim
protection. Before rollback, stop new continuation submissions, coordinate and
drain or cancel pending runs, preserve their checkpoints and delivery receipts,
and separately reconcile external coding jobs. Do not strip transcripts or
restore a pre-cutover database over subsequent user work. No data conversion is
needed for this stored format; rollback behavior still needs a manual rehearsal.

Raw tool output, command lines and authentication diagnostics are discarded by the
supervisor. The final assistant summary is bounded and common credential formats
are redacted; this is not a guarantee against arbitrary sensitive text. Codex's
own session history remains under its home and follows its normal storage behavior.

Validation uses mocked unit tests for process, HTTP and UI handler boundaries.
Signed-in execution, platform package availability, Linux sandbox enforcement and
actual descendant cleanup require an operator's manual sandbox trial. No inference
or service deployment is part of automated validation.
