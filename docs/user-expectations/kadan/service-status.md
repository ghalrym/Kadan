# Service status

Kadan shows connection status for its connected **Kadan Server** in the bottom of the sidebar and above the chat composer when something needs attention. Managing the agent's stored data, local models and subscriptions lives in [Kadan Server](../kadan-server/overview.md); the standalone Kadan client does not link directly to Tsuchi or Hikari pages. The indicators refresh periodically and can lag a change. Loading and connection failures must remain distinguishable from readiness.

Storage availability means Kadan Server can reach Tsuchi. If storage is unavailable, Kadan cannot reliably load agents or confirm new saved messages. An error must not be presented as an empty collection or a successful save.

Language inference can require a user-approved model download or attention to an unavailable worker in **Kadan Server → AI models**. Language readiness does not mean the model is already loaded or that speech, images and other inference types are ready. A later request can still wait or fail.

When the language model needs downloading, Kadan shows a plain status notice above the composer directing the user to **Kadan Server** and keeps the composer available for notes. Replies become available after Kadan observes the completed checkpoint or active subscription in Kadan Server.

When language inference is unavailable, users can [save chat notes](saving-chat-notes.md) if storage is working. Restoring inference does not automatically answer those notes. Readiness also does not prove that an agent has an available [runner](runner-availability.md).

An agent card's last-active label is based on its latest delivered assistant message. It does not prove that the agent is currently computing or that all of its background work has finished. The card's **Call** action starts a [voice call](voice-calls.md).
