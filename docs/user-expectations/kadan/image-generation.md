# Images in agent conversations

Kadan agents can request image generation through Hikari during a text conversation. Submission starts background work; it does not mean an image is ready. Users can continue sending messages while the image waits or generates.

Hikari supplies the artwork, and Tsuchi stores completed images for the agent. Completion wakes the agent to deliver the image in the same conversation. Saved artwork and delivered messages survive navigation and service recovery. A completed generation and delivery to chat are separate steps.

Model loading, downloads, other inference and available memory can delay or prevent generation. Failed work must remain distinguishable from completed artwork. An interrupted upload can still finish saving in Tsuchi. Kadan keeps that work pending across service restarts until Tsuchi confirms whether it saved the artwork. Storage unavailability can delay this confirmation. A late success can then wake the agent to deliver the saved image once; a confirmed failure is reported without submitting another generation. A new request can produce different artwork.

Closing the browser leaves saved image work running. **Stop image** requests cancellation separately from stopping the chat response. Cancellation can remain pending during cleanup or while an upload's outcome is uncertain. Once cancellation is recorded for pending work, a late upload must not cause chat delivery. Stopping does not remove artwork or messages already saved. See [Stopping chat work](stopping-chat-work.md).

The agent's request describes the image; it does not guarantee that the result follows every detail. Users should inspect delivered artwork before relying on it. Hikari's separate [Images playground](../hikari/image-generation.md) has different result retention.

On the subscription-image branch, conversation artwork follows Hikari's independently selected image provider. ChatGPT generation can run without a local image model or local inference memory. Provider errors remain failures, and completed PNGs use the same Tsuchi storage and conversation delivery flow. Existing pending jobs are not automatically resubmitted when the provider changes. This branch behavior awaits owner validation; it is not asserted as verified on master.
