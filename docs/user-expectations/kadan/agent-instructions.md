# Shared agent instructions

Kadan uses a human-maintained instructions file for text agent work. Operators can edit it to change the shared behavior supplied to agents. It is separate from each agent's saved memories and optional [persona](personas.md), and the chat interface does not provide an editor for it.

Instructions are read when a run first prepares its context. Edits affect later runs, including runs started by image completion events. A running or resumed run retains its saved instruction snapshot. Changing the file does not restart work or rewrite conversation history.

All runners should receive the same intended file. Different copies can make behavior depend on which runner handles the agent. An unreadable file or instructions too large for the available context can fail the response; Kadan must not claim that work completed successfully.

Instructions guide the model but do not guarantee obedience, add tools or change enforced command limits, cancellation and delivery behavior. The agent cannot edit this human-maintained file through the feature. Voice calls use their separate conversation flow.

See the [operator instruction-file setup](../../architecture/kadan/agent-conversations.md#shared-instruction-file) for how to supply an edited file.
