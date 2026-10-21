# Saving chat notes

Kadan lets users save text in an agent's chat without asking for a reply. Saving requires Kadan and Tsuchi to be available, but does not require Hikari or a downloaded model.

Choose **Save as note · no reply**. When Hikari is unavailable or no usable model is selected, the main action becomes **Save note**, and Enter saves a note. Making the model ready later does not answer old notes; send a new message when a reply is wanted.

A confirmed note stays in the conversation and appears as an ordinary message. It can influence later replies within the agent's limited recent context, but it does not create a workspace Memory or guarantee future recall.

Saving a note does not stop work already in progress. See [Stopping chat work](stopping-chat-work.md).

Text shown while sending is not confirmation that it was saved. After an error, use **Reload** to check the conversation. **Retry** under **Unsent message** keeps the original text and note mode. Notes cannot be saved while storage is unavailable.

The storage and retry contract is described in [Chat notes](../../architecture/kadan/chat-notes.md).
