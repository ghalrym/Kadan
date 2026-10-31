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

## Asking the computer directly

The bottom of the computer view has a request box. When the agent's computer assistant is idle, sending a request starts a new task for it. While a task is running, a request steers that task: the assistant reads it before its next step and follows it over the original instructions.

The assistant's own working chat is never shown. The view's header shows only whether it is **Idle**, starting or **Working**, whether it has yet to read a steering request, and how many more tasks are queued. For two minutes after a task ends, it shows whether the task was done, failed or stopped, with the assistant's summary, before returning to **Idle**. The request box shows text only when a request cannot be sent. **Stop the current task** cancels it after the action already in progress; that action is not undone. Tasks started from the view report to no conversation. If the agent handed the task over, its conversation still receives the result, including any changes from steering.

A request sent just as a task finishes is never lost: it either steers that task before it ends or starts the next one. Requests and tasks wait while Kadan's runner is unavailable. If Hikari is unavailable when the assistant needs it, the task fails and reports that instead of retrying on its own.

## Agent actions

Agents use the same computer for [browser, file and command actions](agent-actions.md) through their computer assistant. Watching or taking control does not enable a separate copy of their environment. Files on that computer are separate from Tsuchi's stored files unless explicitly [saved into Workspace](file-delivery.md). Agents can also [copy a Workspace file onto the computer](opening-workspace-files.md) without changing the stored original.

A disconnected view offers **Reconnect**. Reconnecting does not undo actions or restart the browser. Some browser failures can require a restart that loses open tabs and page state. A timed-out action may already have happened, so uncertain changes need inspection before retrying.

The [security and operator responsibilities](../security.md) apply. See [Computer architecture](../../architecture/kadan/agent-computer.md) for access and recovery.
