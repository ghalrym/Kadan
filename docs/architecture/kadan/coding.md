# Codex threads

The [coding expectations](../../user-expectations/kadan/coding.md) describe the
user experience. Kadan runs the official pinned Codex CLI inside the shared
computer and treats Codex's own session history as the only record of coding work.
Kadan keeps no job store, so there is nothing to reconcile between Kadan and Codex.

## Computer-side script

The sandbox image installs `@openai/codex@0.162.1` and copies `codex_threads.py` to
`/opt/kadan`. Every operation is one invocation of that script through the sandbox's
Bash exec route, with base64 JSON arguments rather than shell-interpolated text.
`CodingClient` closes each disposable Bash session so polling cannot fill the
sandbox's session table. The script prints fixed error messages; shell stderr and
authentication output never reach the API. `CODEX_HOME` is `/home/gem/.codex` on
the persistent home volume.

- `login` starts `codex login --device-auth` detached and returns the verification
  URL and code parsed from its output. The device flow is used because the user's
  browser cannot reach a localhost callback inside the computer. `setup` reports
  `signed_in` from `codex login status`, `signing_in` while the login process
  lives, and otherwise `signed_out`. `logout` runs `codex logout`.
- `start` launches `codex exec --json` (or `codex exec resume <thread>`) in its own
  session with the prompt on stdin, workspace-write sandboxing, no approvals,
  `gpt-6.1-sol` and high reasoning effort. It returns once Codex emits
  `thread.started`, so the agent gets a thread ID without waiting for the turn.
  A new thread's folder is created if missing, only after sign-in is confirmed.
  Resuming always uses the thread's recorded folder.
- `threads` and `thread` read rollout files under `CODEX_HOME/sessions`. Titles come
  from the first user message; items are user and agent messages plus one-line
  summaries of shell calls and patches. Reads are capped at fifty threads and the
  latest two hundred items.
- `stop` marks the run as stopped, then signals its process group with TERM, then
  KILL.
- `finished` reports exited runs with their outcome, read from the run's JSON
  events: `completed` after `turn.completed`, `stopped` when stopped, otherwise
  `failed`, with Codex's last agent message or error. `acknowledge` deletes a
  reported run.

Run records under `/home/gem/.local/state/kadan/codex/runs` are named by the agent
action's ID and hold the PID, its start time, the thread ID, the folder and the
origin (agent, conversation and run that started it). The action ID makes `start`
idempotent: a runner restart that replays `codex_start` gets the original run back.
Records outlive their process until acknowledged, so a finished run's result
survives Kadan and runner restarts. Threads started outside Kadan appear in the list
but are never reported as running and produce no events.

## Completion events

Starting Codex and reporting its result are separate agent runs, the same pattern
as image jobs. `codex_start` starts the run, then delivers the agent's `message` and
ends the conversation run, so the agent cannot keep working while Codex runs.
The runner's `CodexEvents` loop polls `finished` every five seconds. For each run it
asks Tsuchi to queue an event run in the origin conversation with
`kind: "codex"` and the outcome as content, using a run ID derived from the Codex
run, and only then acknowledges it. A crash between the two repeats the request,
which Tsuchi answers with the existing run. A 404 or 409 (conversation missing or
trashed) drops the result. The event run's context ends with
`Codex thread event: {...}`, and the agent reports the outcome with `send_message`.

## API and agent tools

`GET /api/computer/coding` returns sign-in state and recent threads.
`POST /api/computer/coding/login` and `/logout` change sign-in.
`GET /api/computer/coding/threads/{id}` reads a thread, and
`POST /api/computer/coding/threads/{id}/stop` stops it. The browser cannot start
Codex work; only the agent can.

The agent has three tools. `codex_start` is a conversation tool because it delivers
a message and ends the run; it replays safely through the action-keyed start.
`codex_threads` and `codex_stop` are ordinary computer tools, so after a runner
restart only the GET `codex_threads` repeats and an interrupted `codex_stop` is
reported as uncertain.

Validation uses unit tests with mocked sandbox HTTP and process boundaries. Signed-in
execution, the device-login output format and process-group cleanup need a manual
sandbox trial.
