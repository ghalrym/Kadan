# Transcription API connection

The Speech to Text page now sends its audio reference and formatting preference
through the generated typed `transcribeAudio` client. It supports pending state,
request cancellation, errors, and copying a transcript if a future provider
returns a valid response. The reference is text; this is not a file uploader or
microphone recorder.

**No transcription provider exists yet.** Every valid request currently returns
HTTP 503 with an explicit unavailable explanation. The server does not fetch or
open references, access local files, or produce canned transcripts. Invalid,
blank, overlong, and extra-field inputs are rejected with HTTP 422. Actual speech
to text, audio upload/storage, reference resolution, and formatting implementation
remain future work.

For a manual connection check, run the existing frontend/backend development
services, open Speech to Text, enter any nonblank reference, toggle formatting,
and submit. Expect the unavailable message and no transcript. Disconnecting the
backend should show a connection failure; no fixture appears in either case.
Browser cancellation aborts its HTTP request, not any future provider job.

Controlled frontend tests cover the real SDK request shape, unavailable/error
responses, cancellation, and malformed successes. A synthetic successful response
exercises the future result path only and is never used by production code.
Backend tests cover request validation and the truthful unavailable response.
