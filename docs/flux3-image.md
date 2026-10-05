# FLUX 3 Image API

FLUX 3 Image is a hosted BFL provider, not a downloadable generative checkpoint.
The user supplies `BFL_API_KEY` to the API process. Kadan never creates credentials
or contacts BFL during startup. A generation or edit submits one paid API task per
image; it never automatically resubmits a failed POST. Reference uploads are sent
to BFL as inline PNGs. Output PNGs use the existing local atomic image publisher.

The initial slice uses 1k resolution, the existing aspect/count controls and up to
one uploaded reference. FLUX exposes no seed parameter; explicit seeds are rejected
and result metadata does not invent seeds. The existing page labels the API provider
and disables its seed control. Kadan reserves 1 GiB host RAM for decoding and publishing;
no GPU model is loaded. The provider's moderation defaults are preserved.

Cancellation stops local polling/download/publication. It cannot revoke an already
submitted BFL job or its cost; upstream jobs may continue after a timeout or cancellation.
Polling accepts HTTPS BFL API endpoints only. API credentials are not sent to signed
output URLs, and redirects are not followed. Generation results are downloaded promptly
rather than storing expiring signed URLs in local history.

Official contracts verified 2026-10-05:
- https://docs.bfl.ai/api-reference/utility/generate-an-image-with-flux-3
- https://docs.bfl.ai/api-reference/utility/get-result
- https://docs.bfl.ai/flux_3/flux3_image_overview

Tests use HTTP and tiny image fixtures. No paid API request, credential change or
live model inference was performed. BFL's public endpoint currently supports only
`version=latest`; results are not pinned to an immutable model revision.
