> Historical benchmark record: temperature readings and thresholds below describe the original machine-local procedure, not Kadan runtime policy. Current benchmark temperature monitoring belongs outside the repository; see [external monitoring](/docs/EXTERNAL-BENCHMARK-MONITORING.md).

# Rank-local CUDA blocking-wait candidate

Status: prepared for independent exact-head review. No diagnostic or further image GPU execution has been authorized or run by this change. MR125 remains draft/unmerged; default image backend remains single.

## Evidence and change

Retained c17b340 telemetry: rank main TIDs 2383954/2383955 accumulated 33,257 CPU ticks; all observed helpers 553 ticks (98.36% main, HZ=100). Final 29.75 seconds: 5504 main/132 helper ticks (97.66%). Names and stacks were not captured, so this does not identify the CUDA driver as the cause. Repeated scalar checks, Gloo consensus and optional mask copies are unchanged.

CUDA 13.0.1 documents that `cudaSetDeviceFlags` can update an already-initialized current device, and that blocking synchronization changes host wait behavior. See https://docs.nvidia.com/cuda/archive/13.0.1/cuda-runtime-api/group__CUDART__DEVICE.html . The installed API image contains `/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`.

The rank first binds its Torch device. The helper locates exactly one already-loaded libcudart from `/proc/self/maps`, opens it with RTLD_NOLOAD, checks CUDA major 13, explicitly binds/queries the rank device, then changes only the three scheduling bits to BlockingSync (4). It verifies the complete flag word and device afterward; unrelated bits must be identical. Errors, ambiguous runtime, unsupported version or readback mismatch fail rank startup and retain existing controller teardown. Ready receipts record before/after flags and runtime version. No OS power setting, affinity, numerical operation, NCCL transport or timeout changes.

## Short diagnostic for review

Use two fresh, separately supervised runs: unchanged policy control, then blocking candidate. Use the same image, two physical GPU UUIDs and logical mapping as the retained c17b340 window, both ranks confined to the same CPUs 0,8 from interpreter startup. Match CPU quota/cgroup configuration across cases; do not compare unrestricted placement to pinned ranks. Record the actual cgroup limits, rank PID/start time and every thread name/TID/CPU ticks/affinity, plus stage timestamps and before/after runtime flags.

The first diagnostic should isolate waiting with the bounded `cuda_wait_probe.measure` body: deterministic 2048-square BF16 matrix multiplication, repeated result-scalar synchronization for at most five measured seconds or 4096 iterations per rank. Each case uses fresh child processes, identical operands, TF32 disabled and a 512 MiB per-rank allocator ceiling. Report process CPU/wall and main-thread CPU/wall, iterations/second, sensor peaks, and full output byte SHA256. Require exact output hashes between control and candidate on each physical GPU. This is a wait-policy diagnostic, not image parity or a performance claim about the image pipeline.

Execution wrapper requirements remain mandatory before running the body:

- Independent approval and passing CI for exact source head, full reviewed procedure/settings, unchanged 32 GiB retained-evidence ceiling; reserve at most 16 MiB new evidence including bounded logs and telemetry.
- Queue empty and exact original API identity/config recorded; pause original API, hold existing inference.lock inode through all child teardown. No concurrent text/image/native validation. Preserve model/media data.
- Original cooldown below 60 C for five samples two seconds apart. CPU guard 80 C and GPU guard 90 C stay unchanged, checked externally at least once per second. Cool down separately before each case. Sensor failure aborts.
- 30-second total stage bound per control/candidate, 30-second shared kill/reap cleanup deadline, existing 1800-second API pause/600-second recovery ceilings. Parent-death fencing and process-group/container teardown retain ownership until physical cleanup is confirmed. The five-second loop alone cannot bound a stuck CUDA call.
- Verify both owned rank PIDs disappear, no retained device allocations, exact original API restored and native text ready, queue empty. Preserve failed-run evidence; never retry automatically.

If CPU usage falls and temperatures remain below the unchanged guard, the next independently reviewed diagnostic should replay one retained real cached block with identical tensors and compare all numerical outputs. Only then consider a full image window. Do not infer image parity or thermal success from this microdiagnostic. The probe deliberately has no standalone GPU execution CLI; its default invocation prints the plan and the function must run inside the reviewed external supervisor.

## Concrete execution wrapper

`launch_cuda_wait.py` adapts the established `launch_api_baseline.py` path and directly reuses `restore_exact`, plus `launch_trajectory` identity, physical-owner, thermal and absolute-deadline primitives. It runs one reviewed control or blocking case per exact API pause, with a 30-second host alarm as well as the 30-second PID1 stage limit. `supervisor_cuda_wait.py` holds the existing lock through group kill/reap, and owned children install parent-death SIGKILL before importing Torch. Cleanup signals both groups before waiting under one deadline; the host's single 30-second cleanup deadline includes container stop, log export, removal and physical-owner confirmation. Exact restoration also checks the FIFO empty. No new queue, service or general supervisor framework is introduced.

Default invocation is read-only. For review, the executable form is `python3 docs/real-weight-denoiser/launch_cuda_wait.py --execute-reviewed HEAD --case control --review-record REVIEW.json --evidence NEW_DIRECTORY --retained-roots ROOTS...`, then the separately approved blocking case after full restore and cooldown. Review JSON must match the exact head/protocol/case/settings/criteria emitted by the default invocation. Both cases require CI and clean exact source. Compare their rank JSONs using `cuda_wait_probe.compare`. It now requires identical verified physical UUIDs and returns inconclusive if the control already used BlockingSync; candidate readback must show BlockingSync. No execution has occurred.

## Watchdog and evidence correction

Thermal checks now run in `cuda_wait_monitor.ThermalWatch`, independently of Docker and task telemetry. It targets 500 ms samples, rejects a sample gap above one second, and bounds each GPU query to 350 ms. A missing/slow sensor or threshold failure sends SIGUSR1 to interrupt the main launcher's slow command immediately and enter its existing single-deadline cleanup/restoration path. CPU80/GPU90 limits remain unchanged. The watchdog is stopped during controlled teardown, before restoration; teardown errors cannot skip child cleanup.

The host records actual cgroup `cpu.max`, `cpuset.cpus`, `cpuset.cpus.effective`, `memory.max`, and `memory.swap.max`. Per-process/thread `/proc` samples include PID/TID, start ticks, name, user/system ticks, current CPU and allowed CPU list, with clock tick frequency. The host 30-second alarm starts before Docker launch. Rank reports re-read flags/device after the measured window. A control already blocking or changed during the window fails before any candidate. Blocking launch additionally requires same-head control results, verified restoration and retained-storage inclusion through `--control-evidence CONTROL_DIRECTORY`; comparison is persisted after candidate completion.
