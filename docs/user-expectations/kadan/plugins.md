# Plugins and T3 Code

Kadan lets you connect external tool services (MCP servers) as **plugins** so your agent can work with them during text conversations. Plugins are set up once on Kadan's **Plugins** page and shared across all chats; Tsuchi stores the saved connections and tool lists. Voice calls do not use plugins.

## Browsing plugins

Open **Plugins** in the sidebar. The page lists **Installed** plugins with their status and an **Available** catalog of plugins you can add. Search filters both lists by name. Choosing a plugin opens its page with its description, status, controls and the tools it provides.

## Connecting T3 Code

Kadan also includes built-in [Work](work.md) threads directly in the sidebar. Connecting an external T3 Code server here remains optional for users who already run T3 Code and want to use it alongside or instead of native Work.

Choose **T3 Code** under **Available**:

1. Paste your T3 Code address and choose **Connect T3 Code**.
2. If T3 Code asks you to sign in or approve access, Kadan opens an approval page in a new browser tab. Complete the approval there; the plugin's status switches to **Connected**. **Open the approval page again** reopens it if the tab was closed. If you opened Kadan from another device over a local network address and the approval tab cannot load back into Kadan, copy the full address from that tab and paste it into Kadan's **Finish sign-in** box.

Choose **Custom plugin** to add any other remote MCP server by entering its name, address and sign-in method (browser sign-in, access token, or no sign-in). Saved tokens are never shown back in the browser after you save them.

## Managing plugins and tools

Each installed plugin's page shows its current status, the tools it provides, and controls to:

- **Pause** or **Turn on** the plugin without deleting its settings.
- **Refresh tools** to reload the list of tools the remote service offers.
- **Sign in** again if sign-in expires or the remote server's settings change, or **Sign out** to forget stored sign-in details.
- **Remove** the plugin and its stored credentials.

To keep conversations fast and reliable, Kadan keeps a small curated set of T3 Code tools ready at all times (listing projects and threads, reading threads, starting or messaging threads, interrupting threads, and answering pending questions). Optional T3 Code tool groups, such as pull requests, CI checks, issues and scheduled tasks, can be switched on under **Always ready** on the T3 Code page. The agent can also search and call any other tool offered by a connected plugin when it needs one.

## How the agent works with T3 Code

When you ask the agent to code, investigate a repository, or follow up on a coding task, it delegates the work to T3 Code and replies in chat with a direct link to the T3 Code thread so you can watch live progress, diffs, approvals, and terminals inside T3 Code itself.

- **Background completion:** When a T3 Code thread that the agent started finishes, fails, or is cancelled, Kadan wakes the agent in the same conversation to report the outcome without waiting for another message from you.
- **Questions mid-run:** If a T3 Code thread pauses to ask a clarifying question, Kadan wakes the agent in chat so it can relay the question to you and pass your answer back to T3 Code.
- **Stopping work:** Chat **Stop response** cancels only Kadan's reply; it does not stop a T3 Code thread already running. Ask the agent to interrupt the thread or stop it directly in T3 Code.
- **Trashed conversations:** Background updates for a conversation that has been moved to trash are dropped.
- **Unavailable services:** If a connected service is offline, paused, or needs sign-in again, the agent sees the error and lets you know to check **Plugins**.

See [Agent actions](agent-actions.md), [Stopping chat work](stopping-chat-work.md), and [security and operator responsibility](../security.md).

## Validation status

End-to-end connection, OAuth browser approval, live T3 Code thread launch, and background wakeup delivery are verified by the project owner's manual testing. Automated tests cover storage, protocol handling, tool binding, and watcher state transitions using mocked boundaries.
