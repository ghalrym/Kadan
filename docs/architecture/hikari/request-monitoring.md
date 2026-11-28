# Request telemetry

Hikari observes generation HTTP handlers separately from worker execution. Kadan Server's [Activity](../../user-expectations/kadan-server/activity.md) presents these records.

Completed HTTP observations are process-local and capped at 1,000. Restart clears them. The UI initially shows 25 rows per page; filtering and searching retained IDs, endpoints or summaries reset pagination. Details hold structural summaries rather than original content or credentials. Monitoring reads do not count as generation requests.

The request model field retains only curated Inkling 2 Mini language IDs (`inkling-2-mini-iq4-nl`, `inkling-2-mini-q8-0`, `inkling-2-mini-abliterated-iq4-nl`, `inkling-2-mini-abliterated-q8-0`). It observes the explicit `model` field of a complete JSON body within the summary limit; absent, arbitrary, oversized or unconsumed input leaves the model null. This records a requested checkpoint, not proof that inference ran. The chat completion endpoint does not accept `model_id` as an alias.

Requests per minute, median HTTP latency and error rate use retained completions from the last 60 seconds. If capacity truncates that window, request count is a lower bound and latency/error statistics describe retained samples. No samples means unavailable statistics, not zero. Active HTTP handlers are not a count of GPU jobs.

HTTP latency includes validation, queueing and cleanup. Streamed-chat time to first token includes HTTP waiting and text buffering; generation-only time to first token excludes that queue. Decode tokens per second excludes prefill and cleanup. An absent field is unavailable, not a substitute for another timing.

Request and detail views poll three seconds after each read completes. Shared-memory meters poll every five seconds and show measured host/device use in GiB, including other processes. Worker reservations are admission estimates rather than those physical readings. Missing probes remain explicit. See [Inference architecture](inference.md) for admission ownership.
