# Images in agent conversations

Kadan agents can request image generation through Hikari during a text conversation. Submission starts background work; it does not mean an image is ready. Users can continue sending messages while the image waits or generates.

Hikari supplies the artwork, and Tsuchi stores completed images for the agent. Completion wakes the agent to deliver the image in the same conversation. Saved artwork and delivered messages survive navigation and service recovery. A completed generation and delivery to chat are separate steps.

Model loading, downloads, other inference and available memory can delay or prevent generation. Failed work must remain distinguishable from completed artwork. After an interrupted request, Kadan checks for saved output; it must report failure when no output can be recovered rather than silently submit another generation. A new request can produce different artwork.

Closing the browser leaves saved image work running. **Stop image** requests cancellation separately from stopping the chat response. Cancellation can remain pending during cleanup, and stopping does not remove artwork or messages already saved. See [Stopping chat work](stopping-chat-work.md).

The agent's request describes the image; it does not guarantee that the result follows every detail. Users should inspect delivered artwork before relying on it. Hikari's separate [Images playground](../hikari/image-generation.md) has different result retention.
