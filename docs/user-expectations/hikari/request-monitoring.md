# Request monitoring

Hikari's Requests page shows recent generation requests with filters, search and individual details. It keeps a bounded history that disappears when the API restarts. Old details can expire; **Retry** cannot recover a discarded record.

The page is a monitor, not a conversation or media library. Details show summaries rather than original prompts, generated content or credentials. **Copy ID** is available; replay and copying a runnable request are currently unavailable.

Request rates, latency and errors describe recent retained observations. They can omit older requests. A missing measurement means unavailable data, and a video's accepted submission can appear before its background job finishes.

Memory meters show observed machine use, including other programs. Request timings can include waiting and cleanup, so they do not measure only time spent generating. Views refresh periodically, and successful monitoring does not mean a model is ready or a generation succeeded.

The monitor does not cancel work. Use the originating playground's controls, then check the result. Losing a monitoring record does not stop its job.

See [Request telemetry](../../architecture/hikari/request-monitoring.md) for retention and measurement definitions.
