# Persistent sensor and step-evidence review

The saved image-A attempt at 9c8353f aborted after a sensor command exhausted its
495.4 ms remaining budget: sample age reached 1.012 seconds. Its 331 successful
queries had median/p95/p99/max 35.8/224.4/367.9/405.2 ms. Six exceeded the former
350 ms command cap and were correctly accepted while still fresh. CPU/GPU peaks
were 71.25/64/68 C, below unchanged limits. No output image or generation timing
receipt completed. Existing evidence and baseline were preserved.

## What was measured, and what remains unknown

Eight bounded read-only comparisons were made with the text API available and
no generation workload. CLI median/max wall time was 48.80/54.23 ms. Popen's
creation-through-exec-handoff median/max was 0.719/0.905 ms. That handoff does NOT
include dynamic linking, NVML initialization, driver queries, formatting or
shutdown, which remain combined in the rest of the CLI duration. NVML library
load took 0.707 ms, initialization 14.06 ms, shutdown 12.28 ms. The four persistent
NVML memory/temperature calls for both GPUs combined had median/max 0.066/9.137 ms.

The paired measurements queried CLI first, so the following NVML reads may have
benefited from warmed driver state. A separate helper-only check took 55.79 ms
including first startup/read, then 0.330 and 0.716 ms for subsequent requests.
These are small idle samples, not latency guarantees under image load. The saved
run did not instrument Popen separately, and cannot establish whether its tail
came from process scheduling, initialization, driver contention or another cause.
Repeated CLI initialization is avoidable overhead; it is not a demonstrated root
cause of the half-second timeout. Other API/host diagnostic CLI probes remain
unchanged in this patch and are possible sources of contention, not proven causes.

## Small monitoring change

The independent ThermalWatch now owns one persistent, read-only NVML helper
process, initialized once with exact physical UUID handles. It imports neither
Torch nor CUDA and allocates no model/GPU buffers. ctypes signatures follow the
installed NVIDIA nvml.h. Memory-v2 fields stay raw: observed `used` already
excludes reserved memory. Free bytes are rounded down to MiB, making the existing
256 MiB minimum conservative. No temperature limit changes (CPU <80 C, GPU <90 C),
no freshness relaxation, and no fallback to stale data are introduced.

The 500 ms target and one-second deadline measured from acquisition start remain.
The guard thread performs no driver/file I/O. IPC responses are bounded to 8 KiB
and checked for sequence, identity and lateness; missing libraries, failed NVML
calls, malformed replies and late samples fail closed. Driver calls execute in
an owned child so a hung call can be killed. Teardown kills/reaps under a bounded
wait; inability to confirm teardown remains a failed run. Linux parent-death
fencing prevents an orphan helper after owner death. As before, software timing
is not a hard real-time guarantee during OS/kernel stalls.

Successful samples record helper startup, initialization, IPC/query wall time,
and each memory/temperature call separately. An interrupted native call cannot
report its completion duration; the parent still records the expired deadline
and sample age. No new image workload validated this change.

## Persisted progress

Ranks already emitted `image_step_returned`, but their supervisor retained stderr
only in memory until stop(). The supervisor now forwards complete, validated
owned-job events to the configured API log sink as each pipe is drained and once
more before returning the final acknowledgement. Partial-line storage is capped
at 512 bytes per rank, diagnostics stay at 8 KiB per rank, and at most 40 sequential
events per rank/request are forwarded. Wrong-job/rank, duplicate, oversized and
out-of-order events are ignored. No prompt or arbitrary stderr is forwarded as a
step event. Existing bounded diagnostic tails remain available at shutdown.

Each event marks a validated host-side transformer return; no GPU synchronization
was added. Already flushed sink records survive API SIGKILL without stop(). An
event still inside the child's pipe at the instant of teardown can be lost; this
is not an exactly-once durable transaction or proof of a completed output image.

CPU tests cover fragmented/bounded/owned progress and a real supervisor killed
without cleanup; persistent helper reuse, driver/identity/protocol errors,
timeout/reaping and parent fencing; and independent freshness abort with a hung
helper, stale reads, evidence failures and existing cleanup ordering.
