# Work threads and coding assistants

Kadan includes a built-in **Work** workspace so you can run, supervise, and organize coding threads directly inside Kadan without setting up an external tool. Tsuchi stores projects, threads, transcripts, and pending requests durably; Kadan drives each coding assistant directly in the chosen project folder.

## Sidebar buckets

The **Work** section in Kadan's sidebar organizes your threads into four buckets:

1. **Default:** Threads that have finished work or need your attention, including threads paused on a question or approval (marked **Needs input**).
2. **Working:** Threads currently running an active turn.
3. **Snoozed:** Threads set aside for a chosen duration or until a chosen time. When the snooze expires, Kadan automatically moves the thread back to **Default** even if your browser is closed. You can also wake a snoozed thread early.
4. **Settled:** Completed or older threads tucked out of the active workspace. The **Settled** section collapses out of the way, can be searched by title or coding assistant, and lets you move any thread back to **Default** with **Unsettle**.

## Setting up projects and coding assistants

Open **Work** setup from the buttons beside **WORK** in the sidebar:

- **Projects:** Add an existing folder on this computer using Kadan's folder picker. Removing a project removes its saved threads from Kadan while leaving your folder on disk untouched.
- **Harnesses:** View availability and connection status for **Codex**, **Claude Code**, **Antigravity**, and **Grok**, and sign in directly from the browser. Saved credentials are never shown back in the browser after you connect. You can sign out at any time.

## Working in a thread

Opening a thread shows its project folder, current status, and coding assistant and model pickers. Every thread runs directly in its project folder.

- **Live progress:** Assistant output, thinking progress, tool actions, and changed files stream into the thread as work runs and remain saved if you reload or restart Kadan.
- **Questions and approvals:** When a coding assistant asks a question or asks permission to run an action, the thread moves to **Default** with an inline card where you can approve, decline, pick an option, or type an answer.
- **Follow-ups and stopping:** While a thread is idle, send a new message to continue the same session. While it is running, you can **Steer** (when the assistant supports live updates), **Queue** a follow-up turn, or **Stop** active work. Stopping cancels active work without undoing edits already made.
- **Deleting a thread:** Deleting a thread removes its saved transcript while leaving your project folder untouched.

## Delegation from chat

When you ask your Kadan agent in **Chat** to investigate a project or make code changes, it can start and manage native Work threads for you and reply with a direct link that opens the thread inside Kadan.

- **Automatic updates:** When an agent-started Work thread finishes, fails, stops, or pauses for a question or approval, Kadan automatically wakes your agent in the same chat conversation to report the result or relay the question.
- **Optional T3 Code plugin:** If you prefer using an external T3 Code server, the [T3 Code plugin](plugins.md) remains available under **Plugins** and can be used alongside or instead of native Work.

## Validation status

Automated unit tests cover durable storage, bucket and snooze transitions, local folder browsing and validation, mocked protocol streams for Codex, Claude Code, and ACP (Antigravity and Grok), replay-safe chat delegation, and sidebar/thread UI behavior. Live end-to-end runs with external coding assistants are verified manually by the project owner.
