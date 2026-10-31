# Stopping chat work

Sending a chat message saves it and queues the agent's response. Closing the browser or navigating away must not cancel that work; reopening the conversation shows its saved messages and work state.

The chat's **Stop response** control cancels the queued and running response work currently shown for that conversation. The interface reports "Response stopped" after cancellation. Messages already delivered remain in the conversation; stopping is not an undo operation.

Chat Stop does not stop [Codex threads](coding.md) already running on the
computer. Stop them from the Coding tab. It also does not stop a task already
handed to the computer assistant; stop that from the [computer view](computer.md).

Once cancellation is recorded, the cancelled run must not deliver another message or execute a saved command. Inference may take until the runner's next ownership check to stop, so the interface must not imply that GPU work ends instantly.

Image generation has its own Stop image control. Stopping a chat response does not cancel an image job already submitted. That job may finish later and wake the agent to deliver its result in the same conversation. To stop both, the user must stop the response and its pending image job separately.

Image cancellation remains visible as "Stopping image…" while cleanup is pending, then "Image cancelled" when complete. A failure must remain distinguishable from a successful cancellation.

A service restart must recover durable work without duplicating messages already delivered. It must not restart a cancelled response. The user can send another message in the same conversation after stopping.

See the [chat behavior and persistence design](../../architecture/kadan/agent-conversations.md) for the supporting service contract.
