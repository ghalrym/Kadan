# Runner availability

Kadan saves conversation work through Tsuchi and uses runners to execute it. Saving or queueing a message does not guarantee that a runner can start it immediately. Hikari separately manages inference availability and hardware waiting.

Each runner has a limit on assigned agents, including idle agents. The default is five. When all runners are full, additional agents and their queued messages remain saved but wait for assignment. Image and pet generation have separate queues; their progress is not a measure of available conversation slots.

Assignments survive restarts. Stopping a runner does not release its agents or automatically move them to another runner. A replacement using the same runner identity can recover ownership after the old session expires. An operator can explicitly reassign an agent to another runner with capacity. Adding a runner alone does not take over a stopped runner's existing assignments.

Recovery can take time. Saved conversations and delivered messages remain, cancelled responses must stay cancelled, and unconfirmed computer actions require inspection before retrying. Releasing an assignment makes the agent eligible for assignment again; it is not a pause control.

Open **Runner assignment** in the selected agent's sidebar to inspect its assigned runner, heartbeat status and occupied slots. **Refresh assignments** reads the current state; the view does not update automatically. A failed read must remain distinguishable from an unassigned agent. An expired heartbeat means the runner has not renewed its availability, not that its process has necessarily stopped.

Choose a runner with a current heartbeat and a free slot, then select **Assign**. **Release assignment** makes the agent eligible for automatic assignment again, possibly to the same runner. Kadan refreshes the assignment after either change. These controls do not guarantee that work starts immediately or that an external action already in progress stops immediately.

If the assignment changes elsewhere or a request fails, Kadan keeps the selected destination and requires a refresh before another attempt. An unconfirmed change may already have been saved. Closing the dialog does not change assignments; while saving, wait for the result before closing.

See the [operator assignment and recovery procedure](../../architecture/kadan/runners.md) and [Stopping chat work](stopping-chat-work.md).
