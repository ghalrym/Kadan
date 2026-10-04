# Image API controls

The Image and Edit pages now send their editable inputs to `/v1/images/generations`
and `/v1/images/edits` through the generated client and load `/v1/images` history.
A request can be cancelled in the browser; this does not promise cancellation of
future provider-side work.

No image inference provider has been implemented. Both POST endpoints return HTTP
503 with a provider-unavailable error, and history is empty. The UI states this
limitation and does not convert metadata into fake thumbnails or success. Editing
currently accepts a source reference only; there is no upload/fetch pipeline.

To check locally, start the API and frontend, enter a prompt, choose an aspect,
count and optional seed, and submit. Expect the explicit unavailable error. On
Edit, also enter a source reference and adjust strength. Network failures and
history retry remain visible. Image model loading, uploads, artifact delivery,
persisted history and integration with the shared resource manager are future
provider work, not completed by these controls.

Validation: `python -m unittest api.tests.routes.v1.images.test_images`, frontend
`npm test`, `npm run lint`, and `npm run build`. No weights or external generation
services are used by these checks.
