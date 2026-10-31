# Coding with Codex

Kadan's text agent can hand coding work to Codex on its computer. Codex uses the
user's ChatGPT subscription and works on that computer's files with GPT-6.1 Sol at
high effort. The computer, its files and the Codex sign-in are shared by every agent
and by the owner. Voice calls do not offer coding tools.

## Signing in

Open **Coding** in the sidebar and choose **Sign in with ChatGPT**. Kadan shows a
sign-in link and a one-time code. Open the link in any browser, sign in and enter
the code. The Coding tab switches to signed in once ChatGPT confirms. No terminal
or desktop access is needed. **Sign out** removes the sign-in from the computer.
Signing in confirms the account, not available usage or model access; a run can
still fail for those reasons. If Codex isn't installed on the computer, the tab says
so and asks the person running Kadan to update it.

## How the agent uses Codex

Ask the agent to work on a project folder on the computer; a new project's folder is
created if it doesn't exist yet. The agent starts Codex in the background and
replies right away with what Codex is doing, then waits for it rather than doing the
work itself. When Codex finishes, fails or is stopped, the agent replies again in the
same conversation with the outcome, without another message from the user. Results
from a conversation that has since been moved to trash are dropped. The agent can
also continue an earlier thread with a follow-up, and can check on a thread when
asked. A thread that is
already working can't take another follow-up until it finishes or is stopped. If
Codex isn't signed in, the agent asks the user to sign in from the Coding tab.

## Tracking threads

The Coding tab lists the Codex threads saved on the computer, newest first, with
each thread's opening request, project folder, last update and whether it is still
working. That includes threads someone started directly on the computer. Opening a
thread shows its conversation: requests, Codex's replies and short notes for the
commands it ran and files it edited. Full command output and diffs are not shown,
and very long threads show only their latest part. The tab refreshes every few
seconds while open.

**Stop** appears on a thread that the agent started and that is still working. It
ends that Codex run; changes it already made to files remain. Chat **Stop response**
does not stop Codex, and closing the browser or restarting Kadan's API or runner
leaves Codex running. Restarting the computer ends any running Codex work.

Threads live in Codex's own history on the computer's persistent home volume.
Removing that volume removes them. Generated replies can contain private project
content; review them before sharing.

See [Agent actions](agent-actions.md), [Stopping chat work](stopping-chat-work.md)
and [security and operator responsibility](../security.md).
