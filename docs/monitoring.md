# Local monitoring

`/v1/requests` and `/v1/metrics` contain actual process-local measurements, not
seed data. The API retains the latest 1,000 completed POST requests to the chat,
decision, image generation/edit, video generation, speech and transcription
endpoints. Reads of monitoring endpoints do not monitor themselves. Other routes
and methods are excluded. Run one API worker; each process has its own history.
History and counters reset when the process restarts; no database records are
inserted. In-progress requests contribute to the active count and enter history
when their handler finishes or is cancelled.

Records contain UTC time, endpoint/type, HTTP status (including validation and
availability failures), elapsed handler time and observed request/response byte
counts. Bodies are never logged. A temporary 16 KiB request buffer produces only
structural summaries such as message counts and character counts. Oversized or
malformed payloads receive a size-only summary. Authorization headers, arbitrary
model names, prompt text, outputs, media and exception details are not retained.
Only explicit small/medium/large model IDs are recorded, as requested model IDs;
these are not evidence of which model actually executed. Token rate and time to
first token are not measured. HTTP telemetry measures request handling; it does not establish model quality
or successful hardware validation.

HTTP duration includes handler validation, generation and cleanup; it is not
model-only latency. Status 499 denotes observed disconnect/cancellation before a
complete response. An unhandled exception records 500 and still propagates through
normal error handling. Response size counts bytes emitted to the ASGI server,
not proof that the browser received them.

Counts, p50 latency and error rate use retained completions in the last 60 seconds.
Empty windows report null latency/error rate. If more requests occurred than fit
the ring, `window_truncated` is true: the count is a lower bound and latency/error
rate represent the retained sample. Cumulative completed and currently active
counts are separate. Filtering precedes pagination, newest first; live arrivals
or eviction can shift offset pages. Expired/restarted request IDs return 404.

Memory meters are measured in GiB. Host RAM is MemTotal minus MemAvailable from
Linux `/proc/meminfo`, including other processes. GPU VRAM is device total minus
free memory via PyTorch only if that optional package is already loaded. Dashboard
polls never import PyTorch or require CUDA; unavailable probes are reported instead
of invented readings. These meters are not Kadan allocation counters.

The UI polls serially, aborts on navigation/filter changes, suppresses late results,
and removes stale values after failures. It provides empty/loading/error states,
filters, search, pagination, and summary-only request details. CPU/HTTP tests use
isolated in-memory stores and controlled handlers; no inference or model download
is needed to validate monitoring.

Validation commands:

```sh
python -m unittest api.tests.routes.v1.test_requests api.tests.services.test_telemetry
npm --prefix frontend run lint
npm --prefix frontend run build
```

Optional browser integration smoke (`frontend/tests/monitoring.browser.cjs`) needs
Playwright and a Chromium installation, plus a freshly started API and Vite proxy.
It creates only failed chat requests, then checks live history, filtering/paging,
privacy, unavailable metrics/history, and stale-response suppression. Set
`MONITORING_TEST_URL` to the Vite URL; `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` can
point to an existing tool installation. It does not install dependencies or load a
model. Run only against a disposable local API process with empty request history.
