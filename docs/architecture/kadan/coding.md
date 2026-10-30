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
request. The owner signs in from the computer's terminal, for example with
`CODEX_HOME=/home/gem/.codex codex login --device-auth`, subject to their account's
device-login permissions. [API-key login](https://developers.openai.com/codex/auth)
is also user provisioned through Codex.
Kadan does not copy credentials from its API host, browser or this development environment.

The API's `CodingClient` and the runner's three tools invoke the same supervisor
through the sandbox's existing Bash exec route. The control invocation receives
base64 encoded JSON rather than shell-interpolated task text, returns structured
JSON, and closes its disposable Bash session so status polling cannot exhaust the
computer's session capacity. Transport failures are uncertain outcomes, never an
automatic retry. Raw shell stderr and auth output do not enter API errors.

`codex_status` reads setup/recent job IDs or one job's events after a cursor.
`codex_start` accepts a caller-chosen UUID request ID, prompt, absolute existing
working directory, deadline and optional previous job ID. The job ID is the
request ID, so lost submissions remain discoverable. `codex_cancel` records a
durable cancellation request. Only status is replayable after conversation-run
recovery; the existing runner fences new actions after chat cancellation.
These tools do not schedule completion wakes. The panel polls without inference,
and the agent can inspect results on a subsequent user request.

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
Raw tool output, command lines and authentication diagnostics are discarded by the
supervisor. The final assistant summary is bounded and common credential formats
are redacted; this is not a guarantee against arbitrary sensitive text. Codex's
own session history remains under its home and follows its normal storage behavior.

Validation uses mocked unit tests for process, HTTP and UI handler boundaries.
Signed-in execution, platform package availability, Linux sandbox enforcement and
actual descendant cleanup require an operator's manual sandbox trial. No inference
or service deployment is part of automated validation.
