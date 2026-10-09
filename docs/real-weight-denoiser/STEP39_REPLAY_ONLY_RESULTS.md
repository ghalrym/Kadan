> Historical benchmark record: temperature readings and thresholds below describe the original machine-local procedure, not Kadan runtime policy. Current benchmark temperature monitoring belongs outside the repository; see [external monitoring](/docs/EXTERNAL-BENCHMARK-MONITORING.md).

# Step 39 replay-only result: passed

Independently reviewed source `03e7ff190ef4e29a7356282153bf5da1d6b270a0` completed
step-39 replay from the retained contexts. Both ranks passed all 172 numerical
audit boundaries, maximum absolute difference **0**, zero violations and zero
nonfinite values. Independent transport ownership/roundtrip checks passed. All 32
teacher blocks, independently advancing 1/4/32-block chains and final norm/projection
were checked with unchanged criteria. No recapture, context deletion or weight copy.

The supervisor exited 0 and reaped all workers after **376.489 seconds**. The exact
API container `1efcf5729f776dd90bbc0611c63a474bc696cadc0d48b9011fbb320e167b6499`
was restored healthy with native `small` ready. Total pause **456.444 seconds**, below
the unchanged 2,100-second limit. API StartedAt: `2026-10-09T00:23:40.031972854Z`.
The original incomplete step-39 attempt and all retained contexts remain unchanged.

## Thermal and CPU evidence

Five fresh cooldown maxima were **50, 48.75, 49.25, 48 and 47.125°C**. Both ranks
recorded intra-op=1, inter-op=1 and cgroup `cpu.max=200000 100000`. Affinity still
contained 32 CPUs: this is an aggregate CPU-time quota, not core pinning.

Across **322 thermal samples**, the recorded maximum was **79°C**, on `k10temp`
**Tccd5**, `/sys/class/hwmon/hwmon3/temp7_input`, at Unix **1791505307.0706575**.
Both ranks' latest phase was `cpu-audit:chain-32-5`, timestamped about 1.4 seconds
earlier. The same sample reported Tctl 74.625°C and Tccd3 54.75°C. Every sample was
accepted under its applicable limit; runtime remained strictly below **80°C**.
These are sampled maxima, not a claim of continuously observed temperature.

First-to-last valid cgroup samples recorded 567.930 CPU seconds, 1,308 throttled
periods out of 3,754 and 6.452 seconds in the cgroup throttled-time counter. The final
empty read during container exit is excluded. Physical GPU peaks were 17,202 MiB
and 16,959 MiB; maximum recorded GPU temperature was 79°C. Cgroup memory.current
peaked at 2,983,563,264 bytes, which excludes some shared cache/whole-host memory.

The successful run confirms these bounds were applied and respected. It does
**not** establish oversubscription as the earlier thermal-stop cause: the original
rejected sample and historical per-rank thread utilization were not recorded, and
cooldown, quota and pool limits changed together. No power/firmware changes occurred.

## Evidence and scope

`evidence-replay-only-03e7ff1/` contains global numerical reports, both rank verdicts,
thread/precision identity, all labelled thermal samples, review and exact-head CI
records, supervisor/container exit evidence and sanitized restoration facts. Raw
container configuration and tensors are not committed.

Exact executed-source CI was green: API push
[37863570031](https://github.com/ghalrym/Kadan/actions/runs/37863570031), native push
[37863570082](https://github.com/ghalrym/Kadan/actions/runs/37863570082). See the saved
CI record for the separately queried PR checks. The source passed 46 CPU tests.

The declared cached-step cases—first slice, step 20 and step 39—have now passed.
This is **not full-image acceptance**. Independent scheduler trajectories, VAE
decoding, decoded-image comparison and API integration remain untested. Prior FP32
gates remain failed, controlled-accumulation v1 remains rejected, and production
activation remains false. See FULL_TRAJECTORY_PROTOCOL_V1.md for the next proposed
review gate; it has not been executed or approved by these results.
