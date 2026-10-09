# CPU budget and thermal attribution follow-up

The failed b57db71 API window recorded 103 safety samples, per-sensor CPU temperatures, GPU process identities/memory, rank PID/session identities and ResourceManager snapshots. It did **not** record per-thread CPU run time, per-thread affinity, process CPU utilization, cgroup CPU counters or host per-core utilization. The recording therefore cannot establish a unique cause for the 80.0 C trip.

Verified observations:

- The rank startup policy selected the first two allowed CPU IDs, configured as [0,1] on this unrestricted host. Both belong to the same highest-level cache domain. Runtime per-thread masks were not captured.
- External sensor peaks were Tctl 78.0 C, Tccd3 79.5 C and Tccd5 65.25 C. At the hottest external sample Tccd3 was 79.5 C and Tccd5 was 50.0 C. The worker captured the rejected 80.0 C reading immediately before cleanup.
- Recorded GPU owners were the established desktop processes, API process, parked native-text context and the two expected image ranks. No additional GPU inference owner was observed. This does not exclude unrelated CPU load.
- The native text lifecycle was offloaded during rank execution. No CPU utilization counters establish how much work the API, text process, CUDA/NCCL helpers or unrelated host processes performed.
- The inherited H3 diagnostic hook was enabled. Its request and output file mtimes remained on October 6, before this October 9 run, consistent with an idle polling helper rather than a triggered tensor scan. No per-thread utilization was recorded, so its exact CPU contribution is unknown.

Interpretation: concentrating two busy ranks on one cache/CCD domain is a plausible contributor to the hot/cool sensor contrast. The data does not directly map sensor labels to Linux cache IDs and does not prove placement caused the trip. Thread oversubscription, unrelated CPU work, and GPU/chassis heat contribution remain unquantified. The spread-affinity retry is a controlled comparison of a shared CPU pair, not proof that each rank occupies a different CCD.

## Aggregate rank bound

Both ranks receive **the same** selected list of at most two allowed logical CPUs. On this host cache-aware selection produces [0,8]. It is not two disjoint CPUs per rank, and it does not add CPU capacity. Torch remains at one intra-op and one inter-op thread per rank; OMP/MKL/OpenBLAS/NumExpr thread settings remain one. Library helper threads may still exist, but they must share the same two CPU execution slots.

One gap was found during this audit: `sched_setaffinity(0, ...)` in Python changes the calling thread; an inherited sitecustomize hook may already have started a helper thread. Startup now uses the pinned runtime's `/usr/bin/taskset --cpu-list ...` before Python executes. All interpreter startup threads inherit the bounded mask. taskset execs Python in the same PID/process group, preserving existing parent-death fencing and cleanup ownership. The worker's own affinity assertion remains as defense in depth. Missing taskset fails startup and follows the existing pair cleanup path; the pinned image was checked and contains it.

27 CPU lifecycle/transport tests pass, including two real children with helpers created before their main functions: main and helper masks match, their union is the one shared set, and its size is at most two. This validates the rank pair's affinity ceiling. The parent API and native-text worker are outside that rank-pair bound; the existing API container has no cgroup CPU quota. Do not describe the entire service as limited to two CPUs.

## Retry evidence and unchanged limits

The prepared bounded retry retains the passing A/B PNGs and original b57db71 source provenance. It records selected masks, per-thread masks and user/system CPU ticks for probe processes, host per-core CPU counters, load average, cgroup CPU counters, GPU ownership and all named CPU sensors. A rank thread outside the selected shared pair aborts validation. These small safety records replace no model work and add no trajectory tensors.

CPU admission remains below 60 C for five samples; runtime CPU remains below 80 C and GPU below 90 C. RAM, per-device memory, 900-second request-group measurement, cleanup and restoration bounds remain unchanged. The same aggregate storage ceiling and reserved evidence caps apply. No GPU retry has run for this follow-up; independent exact-head review and CI remain required.
