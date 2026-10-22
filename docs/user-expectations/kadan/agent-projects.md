# Agent-owned projects

Give a Kadan agent a goal with several steps. The agent can save a plan in
Workspace, work through useful steps during its response and maintain supporting
documents beneath the plan. The plan records the goal, progress, remaining steps,
blockers and decisions. It belongs to that agent and remains separate from its
personal memories.

Project status appears in the plan's tags: active, blocked, awaiting decision,
paused, completed or cancelled. Active means the goal has remaining work; it does
not promise a running background job. The agent should bring progress and exact
blockers or questions to chat and wait when a decision belongs to you. Ask it to
resume after resolving a blocker or decision, or to pause or cancel the project.
Cancelling preserves its plan and documents.

New responses receive a small index of saved plans and can read the current
documents even when the original goal has left recent chat history. The agent
must preserve your edits in Tsuchi. A conflicting edit requires another read;
an uncertain save must be checked before retrying. Changes already saved remain
when a response fails or stops. An in-flight save may finish after Stop.

This first slice uses the existing bounded conversation response. It does not
wake itself to continue, work indefinitely, or guarantee that every goal finishes
in one response. A later message can continue an active plan. Waiting, paused and
terminal plans require explicit direction to resume. Planning quality, meaningful
checkpoints and respecting pending decisions depend on the agent following its
instructions; they are not a separate workflow engine.

Workspace keeps its existing read-only browsing interface; edit documents through
Tsuchi or ask the agent to revise them. Markdown Space documents can be created
and revised by the agent; existing decks can be read. See [Workspace](workspace.md),
[Agent actions](agent-actions.md) and [Stopping chat work](stopping-chat-work.md).
