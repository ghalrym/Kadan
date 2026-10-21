# Viewing and controlling the computer

Open an agent's card from its pet, then choose its computer when Ready. The card's periodically refreshed thumbnail is a screenshot, not a live session. The live view starts in Watching mode: mouse and keyboard input are disabled until Take control is selected. Stop controlling returns to watching. A disconnected view offers Reconnect.

Taking control enables input from that browser view. It does not pause the agent, reserve exclusive control or prevent input from another viewer. Closing the view disconnects the viewer without stopping the computer or cancelling agent work. A new view starts in Watching mode.

In the current prototype, the deployment configures one sandbox computer shared by all agents, even though each card labels it with that agent's name. Switching agents does not create a separate desktop, browser or filesystem. The default Compose setup supplies this sandbox; a separately run Kadan without a configured sandbox reports Not set up. An unreachable configured sandbox reports Unavailable rather than presenting a working computer.

Computer actions depend on the sandbox being available. Some tool results are shortened, and binary results are described by size rather than supplied directly to the agent. A timeout does not prove an action failed to happen. After restart, an unconfirmed state-changing command is not automatically repeated; the agent receives an uncertain outcome to inspect.

For a stalled browser, certain read operations can trigger a browser restart and one retry. That recovery resets open tabs and page state and is reported to the agent. Other uncertain browser operations require inspection before retrying. Reconnecting the live viewer alone does not restart the browser or undo an action.

The existing [security and operator responsibilities](../security.md) also apply to this shared computer.
