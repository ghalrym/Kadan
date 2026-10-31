# Coding with Codex

Kadan's text agent can delegate a scoped coding task to Codex in its existing
computer. Codex works on that computer's files, using GPT-6.1 Sol with high effort.
The computer, files and credentials are shared with the owner and the agent;
this feature does not provide hosted account separation. Voice calls keep their
existing behavior and do not offer coding tools.

Open **Coding** in the sidebar and choose **Sign in with ChatGPT**. Kadan shows a
sign-in link and a one-time code. Open the link, sign in, and enter the code. The
computer shares this sign-in with the agent. **Cancel sign-in** stops a pending
attempt; closing the panel leaves it available when reopened. If an attempt ends
without signing in, start sign-in again for a new code. **Sign out** removes the
computer's saved sign-in; it does not stop an already accepted coding job.
Kadan's API keeps job status and stop controls available if sign-in status cannot
be read, including when the computer needs an update. A signed-in
status does not confirm model access or available usage. If Codex is unavailable,
ask the person running Kadan to update or reconnect the computer.

Ask the agent to work in a project directory on the computer and describe the
change and constraints. Kadan accepts one coding job at a time on the shared
computer. Another submission is rejected while a job is active or cleanup is
unconfirmed. Kadan derives a durable job ID from the agent's saved action;
recovering that action returns the saved job rather than starting it again.
A changed task needs a new submission after the previous job has stopped.

The Coding panel refreshes bounded progress and the final summary while open.
Messages also shows a compact **Codex on shared computer** status, read from the
computer's saved jobs rather than assistant messages or Activity counts. It
distinguishes starting, working, stopping and unconfirmed cleanup from no active
job. The latest finished job is labelled as earlier evidence with its start time;
it is not proof that the current chat request used Codex. The computer is shared,
so this status is not limited to the displayed conversation. **Open Coding** opens
the existing progress, results and separate stop controls. A failed read marks
retained job evidence as last known; unavailable status must not imply idle work.
These updates arrive periodically and can lag changes.

Earlier progress can be omitted, and summaries do not include complete command
output or diffs. A failed refresh labels retained information as last known;
reopening or refreshing reads the saved job without restarting it. Closing the
panel, closing the browser, stopping chat or restarting Kadan's API or runner
leaves an accepted coding job running in the computer.

The agent waits without repeatedly asking its model for progress, receives the
completed or failed result and continues the original task without another user
message. A runner restart reattaches the saved action to its existing job; browser
reconnects do not create another continuation or move it to the currently displayed
conversation. Later messages queue behind the waiting task. Closing the browser
does not suppress its eventual reply.

**Stop response** in Messages cancels the response and suppresses this automatic continuation. It
does not stop the coding process, undo changes or cause a later completion to
revive that cancelled response. The user can send a new message after stopping.
**Stop coding** remains the separate process control. A lost runner assignment,
unavailable inference or exceeded context/command limit can still prevent a reply
and must remain visible as failed or stopped response work.

If the computer cannot confirm a result, Kadan retries status recovery without
inference up to a bounded deadline beyond the coding timeout. It then returns an
unconfirmed outcome to the agent rather than resubmitting the task or waiting
indefinitely. The job can have made changes even when its result is unavailable.

**Stop coding** requests cancellation independently of chat Stop. The panel shows
stopping until Kadan's supervisor confirms its owned processes have ended.
Cancellation, timeout and failure leave edits already made on disk. Jobs have a
deadline of at most one hour; the default is thirty minutes. A successful Codex
turn confirms execution and cleanup, not that every requested change works.
Inspect the files and the reported validation before relying on them.
The project folder is visible beside the job state before the stop control.
State changes are announced to assistive technology; identifiers remain available
under Job details.

If the supervisor disappears or cannot confirm cleanup, the job needs attention
and further coding is blocked. An operator must restart the computer to establish
that its old processes have ended. A computer restart marks unfinished work as
interrupted; it never automatically replays the task. The owner should inspect
partial changes before continuing. A recorded Codex session can be continued with
a new scoped follow-up in the same directory after cleanup is confirmed. In
**Coding**, select its job, enter the follow-up and choose **Continue with Codex**.
Kadan queues that explicit request in Messages, runs it without asking the chat
model to choose a tool, and posts the recorded outcome with its actual job and
session identifiers. A failed, rejected or unconfirmed continuation is reported as
such. This path does not require Hikari conversation inference.

An uncertain submission retains the original follow-up and request identity for
retry in Coding or Messages. Closing the panel leaves accepted work running;
retrying the same request does not create another turn. Finish or discard an
unsent message before submitting a different continuation. Unsubmitted follow-up
drafts survive closing and reopening the panel in the same browser tab, separately
for each conversation and job.

Ordinary Messages remain model-directed: an assistant can omit a requested tool
or make an unsupported claim. Typing a resume request alone does not establish
that a continuation ran. Use **Continue with Codex** and the recorded job outcome
for that confirmation. A saved
session ID does not guarantee that Codex's session files or authentication are
still available; a failed resume reports failure instead of starting a fresh session.

Job records and Codex sessions live on the computer's persistent home volume,
separately from Tsuchi. Removing that volume removes their recovery history.
Progress excludes raw commands, tool output and authentication diagnostics.
Final summaries redact common credential formats, but generated text can still
contain private project content. Review it before sharing.

See [Agent actions](agent-actions.md), [Stopping chat work](stopping-chat-work.md)
and [security and operator responsibility](../security.md).
