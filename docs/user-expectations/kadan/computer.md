# Viewing and controlling the computer

Kadan lets users open the computer from an agent's card when it is ready. The card's thumbnail is a screenshot; opening the computer starts a live view in **Watching** mode.

**Take control** enables mouse and keyboard input. **Stop controlling** returns to watching. Control does not pause the agent or exclude other viewers. Closing the view leaves the computer and agent work running, and a new view starts in Watching mode.

While connected, stopping control or closing the view releases keys held by that view. An interrupted connection cannot guarantee delivery of those releases.

The current deployment shares one computer across all agents. Switching agents does not create a separate desktop, browser or filesystem. Kadan reports an unconfigured or unavailable computer rather than showing it as ready.

Agents can inspect screenshots with the fixed LFM3 vision model when its worker and approved download are available. A failed or incomplete observation reports an error rather than inventing a screen description. Agents can use the same computer for [browser, file and command actions](agent-actions.md). Watching or taking control does not enable a separate copy of their environment. Files on that computer are separate from Tsuchi's stored files unless explicitly uploaded.

A disconnected view offers **Reconnect**. Reconnecting does not undo actions or restart the browser. Some browser failures can require a restart that loses open tabs and page state. A timed-out action may already have happened, so uncertain changes need inspection before retrying.

The [security and operator responsibilities](../security.md) apply. See [Computer architecture](../../architecture/kadan/agent-computer.md) for access and recovery.
