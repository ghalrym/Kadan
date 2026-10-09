# External benchmark monitoring

Kadan does not impose CPU/GPU temperature thresholds, cooldown periods, sensor
inventory requirements or a machine-specific thermal policy. The API's resource
checks enforce admitted host/device memory and physical process ownership.
Request cancellation, deadlines, quarantine and process cleanup remain in place.
GPU driver error checks and hardware/firmware protections are unchanged.

Operators may run separate, machine-local monitoring around benchmarks. That
monitoring and its configuration live outside this repository and are not imported
by the API. Missing temperature sensors must not prevent a normal request.
No environment variable or server setting enables the removed host policy.

Before removing the tracked monitor, its original sources, tests, documentation
and configuration were preserved externally with SHA256 provenance, alongside a
standalone copy of the reviewed monitoring and durable-window tools. Existing
benchmark evidence was not moved, rewritten or deleted. Historical reports in
this repository retain their observed temperatures and failed-run results; their
old thresholds are not instructions for current deployments.

Tracked benchmark helpers now retain memory/ownership/deadline/cleanup checks
without imposing our machine's temperature limits. Execute hardware benchmarks
only under an operator's separately reviewed external procedure. External source,
configuration and evidence must be bound to any future benchmark review; prior
thermal-policy approvals do not authorize a new run.
