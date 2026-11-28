# Image generation

Hikari generates images from a text description for Kadan (including pets and images in chat) and for other apps through its API; Kadan Server has no page for trying it. Requests can choose an aspect, one, two or four images, and an optional seed. A seed is recorded with the result but does not guarantee identical output across models or settings.

Generation uses the image model downloaded from Kadan Server's AI models, or ChatGPT when chosen there. Loading, downloads and available memory can delay or prevent a request. Asking for several images does not promise simultaneous generation.

Progress distinguishes waiting, loading and generation. Percentages and remaining time are estimates. Only completed image results count as success; errors or missing image data must not become substitute artwork.

Hikari returns results to the client and keeps no gallery. Kadan's image-delivery flow stores its artwork through Tsuchi.

Cancelling a request stops waiting and ignores late results. Cleanup can take longer, and an error does not prove that no work happened. Submitting again starts another generation, so clients that save results should check them before retrying uncertain work.

See [Image editing](image-editing.md) for source and reference behavior, and [Image worker architecture](../../architecture/hikari/worker-image.md) for request and cancellation contracts.

## Subscription generation on this branch

This branch adds ChatGPT image generation through the existing Codex sign-in and an independent **Images** choice in Kadan Server's AI models. This is not verified behavior on master; owner validation remains pending. Local uses FLUX.2 Klein 4B with its existing loading and memory requirements. ChatGPT generation needs no local image download or available inference memory.

ChatGPT can generate one, two or four images, requested separately. Aspect is a request to the provider, not a guaranteed output shape; results report their actual dimensions. Seeds and custom sampling steps are unsupported and fail with an explanation. Results have no seed. Only completed PNG artwork is returned; a text response is not artwork. There is no automatic fallback. A failed multi-image request can have consumed usage without returning a gallery.

Subscription limits, account access and provider availability apply. Cancellation waits for Hikari to stop its active request and clean up, but already consumed usage may still count. Disconnecting returns future generation to Local; it does not revoke work already running. Subscription generation does not report native loading progress or estimated percentages. Saved artwork in Kadan follows its existing delivery and cancellation rules.
