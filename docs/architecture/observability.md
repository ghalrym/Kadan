# Observability

The shared `kadan_tracing` package and authenticated native worker trace relay
provide one request context across services. Jaeger is the central view for
request performance: queueing, loading, preparation and execution phases belong
to the same trace. Performance details should enrich an existing span whenever
that span already owns the work. Separate timing logs, log-extraction commands
and parallel profiling workflows should not duplicate that view.

Native workers buffer at most 16 phase records for a sampled request and send
them with its terminal event. The API validates their request context, phase and
timestamps before exporting them. Optional numeric text-phase details use an
explicit schema and export allowlist. Invalid detail is discarded independently
of the phase span and cannot reject a valid inference result. The shared export
boundary removes unknown attributes, events, links and error descriptions.

Text prefill/decode spans carry the existing phase counters and a fixed-size
projection dispatch summary. Native phases can also carry a CUDA summary with
50 fixed custom-kernel categories: launch attempts, host launch seconds,
successfully measured event pairs and GPU event seconds. Template variants and
callsites of the same named kernel are combined. Synchronous `cudaMemcpy` host
seconds and attempts use all five fixed requested copy kinds, including
`Default`; that kind does not infer a transfer direction. Stream-synchronization
host seconds and attempts include failing calls. Host API time is never exported
as GPU execution time.

CUDA collection follows the existing sampled-request context across device
threads and image denoiser lanes. Nested leaf spans own their work rather than
duplicating it on their parent. Shared image batches carry the same batch totals
on each sampled member's existing admission/execute span, identified by
`kadan.cuda.shared_batch_size`; these are shared batch costs, not an exclusive
per-request attribution and must not be summed across member traces. Sampling a
batch member observes the shared work, including work benefiting unsampled
members. It does not change batch membership, order or execution.

Driver events bracket the original custom kernel on its original stream and
current CUDA context. Driver event calls do not consume or overwrite the runtime
last-error slot used by inference. Up to 128 event pairs per device execution
thread may be pending. Existing copies and stream waits query completed pairs;
phase/task completion and an actual device switch discard any pairs still
incomplete. There is no telemetry synchronization, device probing or new CUDA
context. Unavailable event operations and capacity/incomplete drops are explicit
counters; measured-pair counts distinguish partial GPU coverage from launch
attempts. Unconfirmed event destruction disables further event allocation on
that thread, bounding potential resource leakage. Telemetry failures cannot
replace an inference result or its original CUDA error.

The native relay buffers at most 16 phase records per sampled request. Its fixed
CUDA schema has 215 numeric attributes; together with text details and existing
span attributes, at most 243 fit within the fixed 256-attribute export limit.
Invalid CUDA and text detail are discarded independently of each other and the
phase. No prompts, token IDs, tensor names, layer/expert IDs, arbitrary runtime
labels, addresses or per-launch/per-token spans are exported.

Projection durations remain completed-call wall time, including backend source
reads, transfers, validation and existing waits. Kernel event sums describe the
measured custom launches, not device utilization or additive wall time. The
retired interposer's useful kernel/copy/stream-wait measurements belong here;
its arbitrary process-wide phase signals and symbol-table reports do not.
Neither it nor this summary attributes internal cuBLAS/ATen kernels, asynchronous
copy GPU duration, other copy APIs, device waits or event-wait host time.
Sampling, event errors, bounded capacity and unfinished work can leave partial
coverage, which the counters expose. All hardware tiers and placement/adaptive
execution policy remain independent of telemetry.

Operational and error logs remain appropriate for process startup, worker
connectivity, exporter availability and failures that cannot be understood from
a recorded request span. Sampling or an unavailable collector must not make the
system unmanageable. Logs should describe the operational condition without
copying request contents or duplicating performance counters.
