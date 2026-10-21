# Saving chat notes

Users can save text in an agent's chat without asking for a response. Saving a note requires Kadan and Tsuchi to be available; it does not require a selected Hikari tier, a downloaded language model or inference.

When chat is ready, **Send** or **Enter** saves the message and queues a response. Choose **Save as note · no reply** instead to save only the text. When Hikari is offline, a tier has not been selected or the selected model is unavailable, the main action becomes **Save note**, and **Enter** saves a note. **Shift+Enter** inserts a newline. The composer accepts up to 16,000 characters and does not save blank text.

A confirmed note stays in that agent's conversation after navigation or reopening the chat. It appears as an ordinary user message, without a separate note badge. Saving it does not create a Memory in the workspace or attach the selected persona. Making Hikari ready later does not queue a response to previously saved notes; send a new message when you want a reply.

In the current prototype, notes share the conversation's history. A later response can include their text as context, subject to the same recent-history and context-size limits as other messages. Saved text does not guarantee that the agent will recall it in every future response.

Saving a note does not stop a response or image job already in progress. That work may still deliver a result. Use the separate controls described in [Stopping chat work](stopping-chat-work.md) to cancel it.

Text shown while sending is not confirmation that storage succeeded. On an error, use **Reload** to check the saved conversation. An unconfirmed submission remains under **Unsent message**, where **Retry** keeps the original text and note mode; it must not turn the note into a request for a reply. Notes require working storage, so this is not an offline editing mode when Tsuchi is unavailable.
