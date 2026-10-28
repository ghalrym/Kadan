# Request telemetry

Hikari observes generation HTTP handlers separately from worker execution. This keeps an accepted background job from being reported as completed inference. See the privately provided expectations for request monitoring.

Completed HTTP observations are process-local and capped at 1,000. Restart clears them. The UI initially shows 25 rows per page; filtering and searching retained IDs, endpoints or summaries reset pagination. Details hold structural summaries rather than original content or credentials. Monitoring reads do not count as generation requests.

Requests per minute, median HTTP latency and error rate use retained completions from the last 60 seconds. If capacity truncates that window, request count is a lower bound and latency/error statistics describe retained samples. No samples means unavailable statistics, not zero. Active HTTP handlers are not a count of GPU jobs; an HTTP 202 video submission is separate from its terminal job outcome.

HTTP latency includes validation, queueing and cleanup. Streamed-chat time to first token includes HTTP waiting and text buffering; generation-only time to first token excludes that queue. Decode tokens per second excludes prefill and cleanup. An absent field is unavailable, not a substitute for another timing.

Request and detail views poll three seconds after each read completes. Shared-memory meters poll every five seconds and show measured host/device use in GiB, including other processes. Worker reservations are admission estimates rather than those physical readings. Missing probes remain explicit. See [Inference architecture](inference.md) for admission ownership.
