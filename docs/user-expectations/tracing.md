# Performance tracing

Kadan, Tsuchi and Hikari can supply request timing traces to an optional Jaeger interface. Tracing is disabled by default. Starting Jaeger alone does not enable collection; operators must also enable application tracing. The applications can start and complete requests without a collector.

Traces help inspect time spent waiting, loading models, executing agent actions and running inference across services. Sampling and bounded buffers mean some requests or phases can be absent. Native worker phases can appear only after work finishes and can be lost on disconnection. A missing trace does not mean that no work happened or that it was fast.

Traces do not measure browser rendering or every hardware operation. Overlapping or shared batch intervals must not be added together as exclusive per-request GPU time. Clocks on different computers can affect how intervals line up. Trace export failures must not turn a successful request into an inference failure.

The baseline Jaeger store is temporary memory with a bounded trace count. It has no promised retention period: old traces can be evicted, and stopping, restarting or recreating Jaeger loses them all. Export a trace from its interface before stopping if it needs to be kept. Traces are not included in Tsuchi backups.

Application tracing excludes prompts, generated content, file contents, tool arguments and results, and credentials. It records approved operational names, timings and measurements. Exported trace files still need appropriate handling, and the interface has no login. The [operator security responsibilities](security.md) apply.

See the [tracing setup](../../README.md#local-tracing-with-jaeger) and [observability architecture](../architecture/observability.md) for configuration and measurement boundaries.
