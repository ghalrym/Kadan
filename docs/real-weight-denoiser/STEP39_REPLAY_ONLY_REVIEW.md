# Step 39 replay-only extension — awaiting independent review

Step 20 passed; step 39 remains incomplete after the CPU-temperature guard stopped
its first replay. This extension verifies and reuses the retained step-39 contexts
without another capture. It has not been executed on GPUs. The restored API stays
available during development and CPU-only verification.

## Thermal investigation

The original launcher monitored the maximum across all `k10temp` and `coretemp`
sensors and required that maximum to be strictly below **80°C**. This host currently
exposes `k10temp` Tctl (`temp1_input`), Tccd3 (`temp5_input`) and Tccd5 (`temp7_input`)
under `/sys/class/hwmon/hwmon3`. These paths can change after reboot; discovery uses
the driver and records each path and label.

The original guard assertion covered both an empty sensor list and a reading at
or above 80°C. It did not retain the rejected sample. Therefore the exact trigger
sensor and peak cannot be reconstructed. Saved maxima were 73.75°C during capture
and 75.75°C during replay; these exclude the rejected sample and are not the true
run peak. Replay was in the teacher-block phase, with the last completed audit
`teacher-vs-capture-19`; exact per-rank activity at the trigger was not recorded.
The retained supervisor confirms termination and reaping, not a numerical failure.

At the investigation's latest saved cooldown sample (Unix 1791504653.776), Tctl
was 49.75°C, Tccd3 45.75°C and Tccd5 56°C. The API was healthy and available.
See `evidence-replay-only-preflight/cooldown-sensors.json` for raw labelled values.
No power, clock, firmware or fan settings were changed.

## CPU concurrency evidence and bound

A no-GPU, no-workload probe used the same pinned image, original four-CPU cgroup
quota and OMP_NUM_THREADS=MKL_NUM_THREADS=1. PyTorch reported intra-op **1**, inter-op
**16**, affinity **32 CPUs**, cgroup `cpu.max=400000 100000`. Explicit setters reduced
both pools to one. This reproduces defaults; historical per-rank counts were not
recorded. Pool capacity does not prove active oversubscription. Eager PyTorch may
not exercise the inter-op pool, and the two ranks' FP64 CPU audit/histogram work can
produce CPU load even with intra-op one. Oversubscription as the cause is **unproven**.

The replay wrapper sets and records intra-op=1 and inter-op=1 per rank before model
or metric operations; OMP/MKL/OpenBLAS/NumExpr are bounded to one. The whole container
has a two-CPU time quota instead of four, verified by the wrapper. This caps aggregate
CPU time; it does not pin cores or promise lower peak temperature. Cgroup cpu.stat
samples record usage and throttling. Extra runtime is possible: the unchanged
900-second stage cap and recovery deadline still terminate an overlong attempt.

Before any API pause, five samples two seconds apart must each be below 60°C.
The runtime threshold remains strictly below 80°C. New thermal JSONL records are
written before rejection, with UTC-compatible epoch timestamps, driver/path/label,
all sensor readings, peak, threshold and each rank's most recent workload phase.
Missing/unreadable sensors are recorded and rejected. Phase timestamps expose stale
samples rather than claiming an exact synchronized view.

## Input identity and storage

`launch_replay_only.py` defaults to a read-only plan. An execution review record must
bind protocol `bf16-step39-replay-only-v1`, the exact new source commit and retained
context manifest `9ceff1294a8aecf4a3211ab952d5b8f3beeec151fc8640431d536e2257c0fafa`.
It requires green exact-head API/native CI and independent bounded-execution approval.

The host verifies all immutable base and retained context hashes, original capture
source `f0b4eefc615a6370fb6bb43a13264a0128e0ea13`, settings, ownership, both prior
incomplete rank identities, prior restoration, and passed step-20 admission. Ranks
verify packets again. The replay source and original capture source are separate
provenance fields. Contexts and weights are mounted read-only. No capture command,
context writes or context deletion occur; the prior interrupted attempt remains
unchanged even if replay succeeds. Only a fresh evidence directory is writable.

Read-only verification passed against the actual retained inputs. Base capture,
step-20 evidence and the entire retained step-39 attempt total **31,747,694,822 bytes**.
Adding the 536,870,912-byte evidence reserve leaves **2,075,172,634 bytes** below the
fixed 34,359,738,368-byte limit. No weight/context duplication is planned.

## Unchanged acceptance and recovery

The same numerical engine checks independent transport ownership, all 32 full-shape
teacher blocks, independently advancing 1/4/32-block chains and final norm/projection.
Atol=rtol=0.02, zero violations and zero nonfinite values remain unchanged. The only
engine edits are optional phase telemetry and separate original-capture provenance.
Production Ulysses arithmetic is unchanged and its source hash is still enforced.

The existing inference-lock inode, 128-GiB RAM/no-swap bound, 22-GiB per-GPU allocator,
32-GiB active artifact limit, thermal/GPU/free-memory guards, 120-second collective
timeout, 2,100-second overall pause and 600-second recovery reserve remain. Exact
container/config/mount restoration and native readiness checks are retained. Only
the owned diagnostic process group/container may be stopped. No production activation.

CPU tests cover default nonexecution, review/source/context binding, rejected sample
persistence, missing sensors, sensor identity, CPU quota rejection and deadline
reserve, alongside the existing numerical/cleanup contracts. All 46 tests pass in
the pinned CPU-only container. GPU replay and runtime fit remain unverified pending
independent source review. Full trajectory/image acceptance is still outstanding.
