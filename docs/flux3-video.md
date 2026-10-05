# FLUX 3 Video API

This provider implements hosted text-to-video with synchronized audio through
`POST https://api.bfl.ai/v1/flux-3-video`. It is not a local generative checkpoint;
FLUX 3 Action's robotics weights are not substituted for video generation.
The user supplies `BFL_API_KEY` to the API process. Startup never calls BFL.

The existing video job queue owns submission, cancellation, failure states and
atomic MP4 publication. The API provider loads no GPU model. This slice accepts
5–20 whole seconds, 24 fps, the existing three aspect ratios and 720p/1080p output.
Provider controls also map 1440p/2160p for future route exposure. Negative prompts
and seeds are rejected because BFL exposes neither. Native providers retain their
previous default seed of 42. The UI labels the API provider and disables unsupported
controls. Input conditioning and draft-enhance are not part of this slice.

Each generation submits exactly once. Polling and downloads can be cancelled locally;
BFL may continue an accepted generation and charge for it. There is no automatic
paid retry or upstream cancellation claim. No API key is forwarded to signed output
URLs. The existing local publisher retains completed output rather than expiring URLs.

Verified official sources (2026-10-05):
- https://docs.bfl.ai/api-reference/utility/generate-a-video-with-flux-3
- https://docs.bfl.ai/api-reference/utility/get-result
- https://docs.bfl.ai/flux_3/flux3_overview

Tests use mocked HTTP and tiny MP4-header fixtures. Real BFL generation, playable
media encoding, cost and latency have not been tested. No paid API calls or credential
changes were made. The public API currently supports `version=latest` only.
