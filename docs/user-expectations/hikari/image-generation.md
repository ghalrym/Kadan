# Image generation

Hikari's Images page generates images from a text description. Users can choose an aspect, one, two or four images, and an optional seed. A seed is recorded with the result but does not guarantee identical output across models or settings.

Generation uses the image model selected through Settings. Loading, downloads and available memory can delay or prevent a request. Asking for several images does not promise simultaneous generation.

Progress distinguishes waiting, loading and generation. Percentages and remaining time are estimates. Only completed image results count as success; errors or missing image data must not become substitute artwork.

The gallery and form exist only in the open page. Leaving, reloading or switching between Generate and Edit loses them. Save required images while available. Kadan's separate image-delivery flow stores its artwork through Tsuchi.

**Cancel request** stops waiting in the browser and ignores late results. Leaving the page does the same. Cleanup can take longer, and an error does not prove that no work happened. Submitting again starts another generation, so clients that save results should check them before retrying uncertain work.

See [Image editing](image-editing.md) for source and reference behavior, and [Image worker architecture](../../architecture/hikari/worker-image.md) for request and cancellation contracts.
