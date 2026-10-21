# Request monitoring

Requests shows observations of generation HTTP requests handled by the current Hikari API process. The default view includes all types, newest first, with 25 rows per page. Filters and the ID, endpoint or summary search apply to retained records and reset pagination. Rows can shift as requests finish or older observations are evicted.

The API retains at most 1,000 completed HTTP observations in memory. Restarting the API clears them. A request detail link can therefore expire or return not found; Retry cannot restore an evicted observation. This is an operational sample, not a durable conversation, saved media library or complete request archive. Details contain structural summaries rather than original prompts, generated content or credentials. Copy ID is available; Copy as cURL and Replay request are currently disabled.

Requests per minute, median HTTP latency and error rate cover retained completions in the last 60 seconds. When history capacity truncates that window, the request count is a lower bound and latency/error statistics cover only retained samples. Missing latency or error statistics mean no samples, not zero latency or errors. Active HTTP requests counts handlers, not GPU jobs. A video can receive HTTP 202 while its background job is still queued or running; its job outcome is shown separately when available.

HTTP latency includes validation, waiting and cleanup. Streamed-chat time to first token includes the HTTP queue and text buffering; generation-only time to first token excludes that queue. Decode tokens per second excludes prefill and cleanup. An absent value means the measurement is unavailable. These timings are not interchangeable with exclusive GPU execution time.

Requests and detail views poll every three seconds after each read completes. Shared memory meters poll every five seconds and report observed host or device usage in GiB, including other processes. Worker reservations are admission estimates, not the same measurement. A missing probe is reported explicitly. Online or a successful monitoring read does not establish model readiness or successful inference, and monitoring reads do not increase generation-request counts.

The monitor does not cancel or replay work. Use the originating playground's controls or the relevant job API for cancellation, then inspect that operation's actual outcome. Clearing or losing an HTTP observation does not cancel its underlying job.
