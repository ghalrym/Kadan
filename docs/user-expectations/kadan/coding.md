# Coding with Codex

Kadan's text agent can delegate a scoped coding task to Codex in its existing
computer. Codex works on that computer's files, using GPT-6.1 Sol with high effort.
The computer, files and credentials are shared with the owner and the agent;
this feature does not provide hosted account separation. Voice calls keep their
existing behavior and do not offer coding tools.

Open **Coding** in the sidebar to inspect setup and recent jobs. Installation and
sign-in are separate: the computer needs the supported Codex package and credentials
provisioned by its owner. A ChatGPT browser session or subscription does not sign
the computer into Codex. The panel explains terminal sign-in; never paste
credentials into chat. A signed-in status reports local credentials, not verified
model access, available credit or a successful coding run. Missing packages,
incompatible versions and unavailable computers remain distinguishable from readiness.

Ask the agent to work in a project directory on the computer and describe the
change and constraints. Kadan accepts one coding job at a time on the shared
computer. Another submission is rejected while a job is active or cleanup is
unconfirmed. The agent receives a durable job ID; retrying an uncertain submission
with that same ID and instructions returns the saved job rather than starting it
again. A changed task needs a new submission after the previous job has stopped.

The Coding panel refreshes bounded progress and the final summary while open.
Earlier progress can be omitted, and summaries do not include complete command
output or diffs. A failed refresh labels retained information as last known;
reopening or refreshing reads the saved job without restarting it. Closing the
panel, closing the browser, stopping chat or restarting Kadan's API or runner
leaves an accepted coding job running in the computer. Completion does not
automatically start another agent response: ask the agent to read the result.

**Stop coding** requests cancellation independently of chat Stop. The panel shows
stopping until Kadan's supervisor confirms its owned processes have ended.
Cancellation, timeout and failure leave edits already made on disk. Jobs have a
deadline of at most one hour; the default is thirty minutes. A successful Codex
turn confirms execution and cleanup, not that every requested change works.
Inspect the files and the reported validation before relying on them.

If the supervisor disappears or cannot confirm cleanup, the job needs attention
and further coding is blocked. An operator must restart the computer to establish
that its old processes have ended. A computer restart marks unfinished work as
interrupted; it never automatically replays the task. The owner should inspect
partial changes before continuing. A recorded Codex session can be continued with
a new scoped follow-up in the same directory after cleanup is confirmed. A saved
session ID does not guarantee that Codex's session files or authentication are
still available; a failed resume reports failure instead of starting a fresh session.

Job records and Codex sessions live on the computer's persistent home volume,
separately from Tsuchi. Removing that volume removes their recovery history.
Progress excludes raw commands, tool output and authentication diagnostics.
Final summaries redact common credential formats, but generated text can still
contain private project content. Review it before sharing.

See [Agent actions](agent-actions.md), [Stopping chat work](stopping-chat-work.md)
and [security and operator responsibility](../security.md).
