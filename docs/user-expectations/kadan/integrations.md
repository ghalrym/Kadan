# Integrations and T3 Code

Kadan lets you connect external tool services (MCP servers) so your agent can work with them during text conversations. Connections are configured once in Kadan's **Integrations** panel and shared across conversations; Tsuchi stores the saved connections and tool lists. Voice calls do not use integrations.

## Connecting T3 Code

Open **Integrations** in the sidebar to set up T3 Code for coding work:

1. Paste your T3 Code server address and choose **Connect T3 Code**.
2. If T3 Code asks you to sign in or approve access, Kadan opens an approval page in your browser. Complete the approval there, and the Integrations card switches to **Connected**.
3. If your T3 Code server uses a bearer token instead of browser sign-in, choose the token option and paste the token in the same form.

You can also add any other remote MCP server by entering its name, address, and sign-in method (browser sign-in, bearer token, or no sign-in). Saved tokens are never shown back in the browser after you save them.

## Managing connections and tools

Each connected integration shows its current status, how many tools it provides, and controls to:

- **Enable or pause** the integration without deleting its settings.
- **Sync tools** to refresh the list of tools the remote service offers.
- **Reconnect** if sign-in expires or the remote server's settings change.
- **Remove** the integration and its stored credentials.

To keep conversations fast and reliable, Kadan keeps a small curated set of T3 Code tools directly ready at all times (listing projects and threads, reading threads, starting or messaging threads, interrupting threads, and answering pending questions). Optional T3 Code tool groups — such as pull requests, CI checks, issues, and preview or device tools — can be turned on from the T3 Code card whenever you want them immediately ready, and the agent can also search and call any other tool offered by a connected integration on demand.

## How the agent works with T3 Code

When you ask the agent to code, investigate a repository, or follow up on a coding task, it delegates the work to T3 Code and replies in chat with a direct link to the T3 Code thread so you can watch live progress, diffs, approvals, and terminals inside T3 Code itself.

- **Background completion:** When a T3 Code thread that the agent started finishes, fails, or is cancelled, Kadan wakes the agent in the same conversation to report the outcome without waiting for another message from you.
- **Questions mid-run:** If a T3 Code thread pauses to ask a clarifying question, Kadan wakes the agent in chat so it can relay the question to you and pass your answer back to T3 Code.
- **Stopping work:** Chat **Stop response** cancels only Kadan's reply; it does not stop a T3 Code thread already running. Ask the agent to interrupt the thread or stop it directly in T3 Code.
- **Trashed conversations:** Background updates for a conversation that has been moved to trash are dropped.
- **Unavailable services:** If a connected service is offline, paused, or needs sign-in again, the agent sees the error and lets you know to check **Integrations**.

See [Agent actions](agent-actions.md), [Stopping chat work](stopping-chat-work.md), and [security and operator responsibility](../security.md).

## Validation status

End-to-end connection, OAuth browser approval, live T3 Code thread launch, and background wakeup delivery are verified by the project owner's manual testing. Automated tests cover storage, protocol handling, tool binding, and watcher state transitions using mocked boundaries.
