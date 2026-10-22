# Runner availability

Kadan saves conversation work through Tsuchi and uses runners to execute it. Saving or queueing a message does not guarantee that a runner can start it immediately. Hikari separately manages inference availability and hardware waiting.

Each runner has a limit on assigned agents, including idle agents. The default is five. When all runners are full, additional agents and their queued messages remain saved but wait for assignment. Image and pet generation have separate queues; their progress is not a measure of available conversation slots.

Assignments survive restarts. Stopping a runner does not release its agents or automatically move them to another runner. A replacement using the same runner identity can recover ownership after the old session expires. An operator can explicitly reassign an agent to another runner with capacity. Adding a runner alone does not take over a stopped runner's existing assignments.

Recovery can take time. Saved conversations and delivered messages remain, cancelled responses must stay cancelled, and unconfirmed computer actions require inspection before retrying. Releasing an assignment makes the agent eligible for assignment again; it is not a pause control.

The current browser interface does not manage runner assignments. See the [operator assignment and recovery procedure](../../architecture/kadan/runners.md) and [Stopping chat work](stopping-chat-work.md).
