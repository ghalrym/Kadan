# Service status

Kadan shows connection information for Tsuchi and Hikari and provides links to their management interfaces. The indicators refresh periodically and can lag a change. Loading and connection failures must remain distinguishable from readiness.

Tsuchi availability means Kadan can reach storage. If storage is unavailable, Kadan cannot reliably load agents or confirm new saved messages. An error must not be presented as an empty collection or a successful save.

Hikari can require a hardware tier, a model download or attention to an unavailable inference backend. Open Hikari Settings to resolve those states. Language readiness does not mean the model is already loaded or that speech, images and other inference types are ready. A later request can still wait or fail.

When language inference is unavailable, users can [save chat notes](saving-chat-notes.md) if storage is working. Restoring inference does not automatically answer those notes. Readiness also does not prove that an agent has an available [runner](runner-availability.md).

An agent card's last-active label is based on its latest delivered assistant message. It does not prove that the agent is currently computing or that all of its background work has finished. The card's **Call** action is currently unavailable; spoken conversations use the chat's [Start call](voice-calls.md) control.
