# Work threads and coding assistants

Kadan includes a built-in **Work** workspace so you can run, supervise, and organize coding threads directly inside Kadan without setting up an external tool. Work runs coding assistants and browses project folders **on the computer where you open the Kadan desktop app**, while Tsuchi on your Kadan Server stores projects, threads, transcripts, and pending requests durably (and Hikari on your Kadan Server powers Inkling 2 Mini inference). Keeping Kadan Desktop open allows local coding threads—including threads started by your chat agent—to execute on your computer.

## Sidebar buckets

Work threads appear in Kadan's sidebar in the same layout as T3 Code: your active threads at the top, with collapsible **Working**, **Snoozed**, and **Settled** shelves below them.

1. **Default (active list):** Threads that have finished work or need your attention, newest first. Each row shows the project, how long ago it changed, the thread title, its status (such as **Needs input**, **Done**, or **Failed**), and the coding assistant's icon. Threads paused on a question or approval are marked **Needs input**.
2. **Working:** Threads currently running an active turn. The shelf only appears while something is running and starts expanded.
3. **Snoozed:** Threads set aside for a chosen duration or until a chosen time, soonest to wake first, each showing how long is left. When the snooze expires, Kadan automatically moves the thread back to the active list even if your browser is closed. You can wake a snoozed thread early from its row or from the thread. The shelf only appears when something is snoozed.
4. **Settled:** Completed or older threads tucked out of the active workspace. The shelf is always shown with its count, starts collapsed, and loads older threads in pages with **Show more**. Move a thread back to the active list with **Un-settle** on its row or **Unsettle** in the thread.

Each shelf shows a count and opens or closes with a click; Kadan remembers which shelves you left open on that browser. Hovering an active thread offers **Settle** without opening it.

The **Search** box at the top of the sidebar filters every bucket by thread title, project, or coding assistant, and opens any shelf that has matches.

## Starting a thread and managing settings

Click the **New thread** button (pencil) at the top of the sidebar to open a new coding thread directly—no title or setup form required. The page asks "What should we build in *project*?" with the composer below it. Choose a project folder and model right there and send your first message; Kadan automatically names the thread from your prompt.

- **Projects:** The project picker under the composer lists your saved folders. Choose **Add folder…** (opening your computer's native folder picker in Kadan Desktop) or **Manage projects**, or manage saved folders under **Settings → Projects**. Removing a project removes its saved threads from Kadan while leaving your folder on disk untouched.
- **Model picker:** The model button in the composer opens a picker laid out like T3 Code's:
  - A column of coding assistant icons on the left, with **Favorites** at the top. Assistants that are not ready are dimmed; selecting one explains what is needed and offers **Open settings**.
  - A searchable model list in the middle. Search covers every assistant. Star a model to add it to **Favorites** (remembered on that browser). The first nine models can be picked with **Ctrl+1**–**Ctrl+9**, and the arrow keys and **Enter** work too.
  - A **Reasoning** column on the right listing the levels that model supports (for example Low, Medium, High), with the default level labelled. Choosing a level selects that model with that level. Models without reasoning levels say so.
  - A footer showing whether the assistant is ready, with **Manage assistants**.

  The chosen model and reasoning level are saved with the thread and remembered for your next new thread. The model cannot be changed while a turn is running.
- **Coding assistants:** Manage connections under **Settings → Coding assistants** for **Inkling**, **Codex**, **Claude Code**, **Antigravity**, and **Grok**.
  - **Inkling** runs through Kadan Server's built-in Inkling 2 Mini inference engine and needs no external account or API key—once Inkling 2 Mini is downloaded on Kadan Server, it is ready to use immediately.
  - External assistants can be signed in to directly from Settings. Saved credentials are never shown back after you connect, and you can sign out at any time.

## Working in a thread

Opening a thread shows its title, project, and status at the top, with **Snooze**, **Settle** (or **Unsettle**), and **Delete** buttons. The activity stream fills the page, and the composer with the model picker sits at the bottom with the project folder beneath it. Every thread runs directly in its project folder.

- **Live progress:** Assistant output, thinking progress, tool actions, changed files, and step diagnostics stream into the thread as work runs and remain saved if you reload or restart Kadan.
- **Screenshots and mockups:** In **Inkling (Local)** threads, you can attach PNG or JPEG screenshots with the paperclip button in the thread composer so the assistant can inspect UI issues or visual mockups without an external vision service.
- **Questions and approvals:** When a coding assistant asks a question or asks permission to run an action (such as a potentially risky shell command), the thread moves to **Default** with an inline card where you can approve, approve for the session, decline, pick an option, or type an answer.
- **Long threads and memory limits:** When a long **Inkling (Local)** thread approaches the active local memory window, Kadan explicitly summarizes earlier steps into a working-memory checkpoint (keeping the goal, modified files, verified checks, open blockers, and next steps) rather than silently dropping messages. If a single request still does not fit available local memory or the active context limit, Kadan stops cleanly and explains what to do next instead of truncating or switching models.
- **Follow-ups and stopping:** While a thread is idle, send a new message to continue the same session. While it is running, you can **Steer** (when the assistant supports live updates), **Queue** a follow-up turn, or **Stop** active work with the stop button that replaces send. Stopping cancels active work without undoing edits already made.
- **Deleting a thread:** Deleting a thread removes its saved transcript while leaving your project folder untouched.

## Delegation from chat

When you ask your Kadan agent in **Chat** to investigate a project or make code changes, it can start and manage native Work threads for you (using **Inkling (Local)** or any connected external assistant) and reply with a direct link that opens the thread inside Kadan.

- **Automatic updates:** When an agent-started Work thread finishes, fails, stops, or pauses for a question or approval, Kadan automatically wakes your agent in the same chat conversation to report the result or relay the question. Background agent-started Inkling Work turns yield priority to your live chat so normal conversation stays responsive.
- **Optional T3 Code plugin:** If you prefer using an external T3 Code server, the [T3 Code plugin](plugins.md) remains available under **Plugins** and can be used alongside or instead of native Work.

## Validation status

Automated unit tests cover durable storage (including image attachments), bucket and snooze transitions, sidebar shelves and search, the model picker (assistant rail, cross-assistant search, reasoning levels, unavailable assistants), local folder browsing and validation, the Inkling (Local) harness loop (bounded tools, edit mismatch recovery, static prefix immutability, explicit token budgeting and compaction, base-weight and priority routing, mocked Decisions guardrails, multimodal payloads, live steering, and cancellation), mocked protocol streams for Codex, Claude Code, and ACP (Antigravity and Grok), replay-safe chat delegation, and sidebar/thread/setup UI behavior. Live end-to-end runs with local inference and external coding assistants are verified manually by the project owner.
