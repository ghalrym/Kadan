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
projection dispatch summary. At most 28 attributes, including stage/worker and
the prefill cache-reuse flag, fit within the shared 32-attribute limit. Values
are finite nonnegative numbers; no prompts, token IDs, tensor names, layer/expert
IDs, arbitrary labels or per-token spans are exported. Additional projection
collection follows existing request sampling, uses no dynamic grouping, and adds
no CUDA synchronization. Non-expert and routed-expert execution decisions and
inference math remain independent of telemetry.

Projection durations measure completed-call wall time, including backend source
reads, transfers, validation and existing waits. Batch acceptance, rejection,
large-row bypass and host-expert counts describe dispatch at the projection
interface. They are not CUDA kernel timings, and nested or concurrent durations
are not additive. Jaeger does not automatically separate dequantization, SGEMM,
transfers or device synchronization; device-level attribution requires a
separately scoped measurement capability and authorization.

Operational and error logs remain appropriate for process startup, worker
connectivity, exporter availability and failures that cannot be understood from
a recorded request span. Sampling or an unavailable collector must not make the
system unmanageable. Logs should describe the operational condition without
copying request contents or duplicating performance counters.
