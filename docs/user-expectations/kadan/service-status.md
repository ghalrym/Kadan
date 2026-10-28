# Service status

Kadan shows connection information for Tsuchi and Hikari and provides links to their management interfaces. The indicators refresh periodically and can lag a change. Loading and connection failures must remain distinguishable from readiness.

Tsuchi availability means Kadan can reach storage. If storage is unavailable, Kadan cannot reliably load agents or confirm new saved messages. An error must not be presented as an empty collection or a successful save.

Hikari can require a hardware tier, a model download or attention to an unavailable inference backend. Open Hikari Settings to resolve those states. Language readiness does not mean the model is already loaded or that speech, images and other inference types are ready. A later request can still wait or fail.

When no plan is selected, Kadan shows **Select a plan to enable replies** beside the chat composer. **Select a plan** opens Hikari Settings in another tab, where users choose a hardware tier and confirm its downloads. Opening Settings keeps the current draft, persona and any unsent message in Kadan. It does not select a plan or send the draft. Users can keep writing or save a note while deciding; the prompt clears after Kadan detects a selected tier. Downloads or other availability issues can still need attention before replies are available.

When language inference is unavailable, users can [save chat notes](saving-chat-notes.md) if storage is working. Restoring inference does not automatically answer those notes. Readiness also does not prove that an agent has an available [runner](runner-availability.md).

An agent card's last-active label is based on its latest delivered assistant message. It does not prove that the agent is currently computing or that all of its background work has finished. The card's **Call** action starts a [voice call](voice-calls.md).
