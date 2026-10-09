# Local evidence durability proposal (no workload approval)

The 3f5404c single-image window tripped at Unix 1791551821.4595459
(2026-10-09 13:17:01.459546 UTC): Tccd3 80.75 C, Tctl 69.125 C, Tccd5 70.75 C.
The saved watchdog rejected the CPU sample before its next NVML query. This was
not a sensor-freshness failure. Existing 80 C enforcement is unchanged.

## Output provenance and limits

Image UUID 1350859f-89e2-4600-bbf3-a0773aa8a220 matches the recovered Redis result
for job f7b62d84dc7c4274a7c66351d4da0544. Its result.json records the requested
prompt and seed42. The published PNG (5,602,120 bytes) SHA256 is
6df59d86af4143efd1e3e5b419812754bd2b9ed1b10ccd33c3bb2ecac4dbf42b,
identical to frozen baseline A. The manager's publication path creates a fresh UUID,
saves the returned image and metadata, and atomically renames its staging directory;
it does not copy the baseline image.

PNG birth is 13:17:03.154902140 UTC, 1.695356 s after the trip. Its mtime is
13:17:04.543972358 UTC; result.json mtime is 13:17:04.544765643 UTC. File times are
not exact rename timestamps. Generation returns before the manager creates the PNG,
so generation completed before its birth, but missing final timing logs prevent
placing completion precisely before or after the trip. Native return validates
40 transformer steps. That is a source-path implication, not a retained 40-step
execution trace. Redis reports succeeded with status503; neither that mixed record
nor an identical PNG proves a successful HTTP200 window.

The last task-counter sample was 0.457 s before the trip. In its preceding 1.082 s,
rank0 main thread accrued1.07 CPU seconds and rank1 main thread1.08; both had affinity
0,8. GPU allocations had already fallen substantially. This is compatible with
postprocessing but does not identify a precise operation. The host accrued about5.0
busy CPU seconds versus2.681 API-cgroup seconds; asynchronous samples make the
remaining2.319 approximate. Host process identities were not recorded, so that
work cannot be assigned to another task or the operator. All461 cgroup samples
report zero quota throttling; hardware thermal throttling/frequencies were not
recorded. The later offline comparison was rejected at its admission guard before
image decoding. There is no evidence that comparison caused the original trip.

## Why final receipts were lost

The foreground remote operator stored request receipts and Docker logs in RAM.
Cleanup stopped the temporary container, read its logs, removed it, restored the
original API and waited for readiness before exporting those values. The remote
connection was lost, its execution session became unavailable, and final files
were absent. Original API restart is independently established. The operator's
exact termination mechanism is not retained; claiming a particular signal or a
completed readiness receipt would be speculation.

## Narrow fix and review boundary

`durable_evidence.py` appends bounded JSONL with file/directory fsync and no symlink
following. `ImageEvents` accepts only progress, timing and publication messages;
its total file cap is256 KiB, each record8 KiB. A dedicated host bind keeps these
records outside the disposable container. Completed records survive producer death;
a killed append can leave a truncated last line and must be reported as incomplete.
Storage stalls remain possible; the independent host watchdog retains cleanup.

The proposed operator diff writes HTTP receipts immediately in the request thread
and FIFO admission identity separately (4 MiB file cap). A missing receipt is
explicitly unknown, never inferred success. New logging configuration and helper
source must be included in a fresh reviewed hash binding. Existing approved bundles
are immutable. The proposal is retained at task-4/image-a-thermal-audit/operator.diff;
it has no fresh approval or runnable binding and was not executed.

`local_evidence_run.py` uses a named local user service, without a shell or automatic
restart, so remote launcher exit does not own the operator lifetime. Pass review
variables explicitly using /usr/bin/env; do not assume the service inherits them.
RuntimeMaxSec1800 and SIGTERM cleanup grace60 seconds bound the service; the operator
keeps its existing earlier measurement/cleanup deadlines. Unique units prevent
accidental duplicate starts. After disconnect inspect that unit and saved evidence;
never retry generation merely to recover logs. Service-manager failure or host loss
still prevents guarantees of final cleanup; incremental evidence remains useful.

Five CPU tests cover producer SIGKILL after fsync, file/record caps, symlink rejection,
event filtering and service command boundaries. A real two-second CPU-only user
service wrote its receipt after the launcher exited, Result=success, ExecMainStatus=0.
No API edit/restart, model load, GPU work or threshold change was performed.

## Sensor-specific policy proposal, not implemented

Installed kernel6.17.9-76061709-generic k10temp (srcversion
A594DFB9AAE185D37A6B0D8) exposes PCI0000:00:18.3 temp1=Tctl, temp5=Tccd3,
temp7=Tccd5. No separate Tdie or critical-temperature attribute is exposed here.
Installed kernel source was unavailable; labels do not establish calibration or
map Tccd3 to an individual logical CPU. The parent supplied AMD's5955WX Tjmax95 C;
this report does not independently equate that specification with every sensor.

For review: warn when any monitored sensor reaches80 C; abort on the first sample
at85 C or higher, retaining fail-closed missing/stale sensor behavior. Clear warning
only after five samples below75 C; never average away an abort sample. Keep the
existing below60 C admission rule and0.5 s sampling/1 s freshness bound. The85 C
proposal gives a nominal10 C margin below the supplied95 C specification, but sensor
offsets, sampling delay and inter-sample peaks mean it is not a certified junction
margin. Verify model-specific k10temp interpretation before adoption. Until approved,
the implemented cutoff remains80 C for all exposed CPU sensors.
