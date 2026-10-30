# Viewing and controlling the computer

Kadan lets users open the computer from an agent's card when it is ready. The card's thumbnail is a screenshot; opening the computer starts a live view in **Watching** mode.

**Take control** enables mouse and keyboard input. **Stop controlling** returns to watching. Control does not pause the agent or exclude other viewers. Closing the view leaves the computer and agent work running, and a new view starts in Watching mode.

While connected, stopping control or closing the view releases keys held by that view. An interrupted connection cannot guarantee delivery of those releases.

Kadan provides one shared computer for its logical agent. Kadan reports an unconfigured or unavailable computer rather than showing it as ready.

The computer has a desktop with an always-visible bottom taskbar in normal windowed use. **Applications** opens a menu with **Browser**, **Terminal**, **Windows** and **Show desktop**. Browser and terminal also have pinned launchers. Opening an app brings its existing window forward, including a minimized window, or opens it if closed. Repeated launcher clicks during startup do not intentionally create extra windows. If an app does not appear within a short wait, a dialog offers a retry through Applications.

Windows have familiar minimize, maximize, restore and close controls. Users can move and resize them, switch using taskbar entries or Alt+Tab, and use Show desktop to reveal the desktop. Closing the browser's last window leaves its background process available for agent tools; the Browser launcher opens a window again. Agent actions may also open or change browser windows. Fullscreen applications can cover the taskbar; leave fullscreen to reach it again.

When desktop startup finishes, the background has the same dark color after a computer restart without needing a manual reset.

Closing or reconnecting the computer view leaves desktop apps running. The computer's home files and browser profile are retained when its container is replaced using the same persistent home. This does not guarantee that every website keeps a sign-in valid. Replacing or deleting that home loses its files and saved sign-ins.

After a computer replacement, a browser profile left locked by the old instance can prevent the browser from starting. Applications and Terminal remain reachable. An operator must confirm that the old browser has stopped before clearing a stale profile lock; resetting the profile is unnecessary.

The default computer has 2 CPUs, 4 GiB of memory and no GPU allocation. Advanced users can change Docker resource limits; heavier apps need sufficient host resources and may require extra installation. Changing limits requires a computer restart, interrupting open apps and running work. These limits belong to the computer and do not change Hikari's models or resources.

Agents can use the same computer for [browser, file and command actions](agent-actions.md). Watching or taking control does not enable a separate copy of their environment. Files on that computer are separate from Tsuchi's stored files unless explicitly uploaded.

A disconnected view offers **Reconnect**. Reconnecting does not undo actions or restart the browser. Some browser failures can require a restart that loses open tabs and page state. A timed-out action may already have happened, so uncertain changes need inspection before retrying.

The [security and operator responsibilities](../security.md) apply. See [Computer architecture](../../architecture/kadan/agent-computer.md) for access and recovery.
