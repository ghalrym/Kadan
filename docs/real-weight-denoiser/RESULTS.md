# First real-weight gate result: stopped on FP32 violations

Executed reviewed9304851657c8e420f117adcccc14b50610ee48bd after all exact-head
CI passed: API PR37850043699/push37850038805, native PR37850043709/push37850038906.
No thresholds or execution code were changed during the run. MR121/MR122 remain
draft and production split remains disabled.

## Pause and restoration

The pause window was320.491 seconds (5m20.491s), including cleanup and verified
model-ready restoration, below2100 seconds with the recovery reserve intact.
It began2026-10-08T21:59:36Z. Capture supervisor ran202.363s and replay36.539s.
The exact original API container1efcf5729f776dd90bbc0611c63a474bc696cadc0d48b9011fbb320e167b6499
restarted22:03:44.290735994Z and was verified HTTP/Docker healthy and model ready.
Its image remains sha256:eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa;
identity/config/mount checks and unrelated-service ID checks passed. Desktop
contexts were preserved, rank processes reaped, physical memory returned to
baseline before restore, and no host OOM or temperature guard occurred. The
launcher returned failure because the numerical gate failed, not because recovery
failed. The model-ready window is not a precisely sampled HTTP-only outage.

## Capture and exchange

All32 trained blocks and final adaptive norm/projection were captured from the
unchanged eager first cached timestep. Settings: pinned d26bb612 revision,
BF16,2048²,40 configured steps,seed42,red-apple prompt. Captured scaled timestep
was0.9921875. Capture work after loading took184.646s, including synchronous CPU
copies, serialization and hashing; it is not denoiser latency. The32 block files
plus tail/manifest total22,869,681,264 bytes (about21.30 GiB), under32 GiB. The
capture is retained locally and is not committed to Git. Its hashes and scheduler
configuration are in evidence-9304851/capture-manifest.json.

Both ranks verified capture digests and bit-exact exchange round-trips. The first
seven shortened FP32 trained-block comparisons (indices0–6) passed on both ranks.
Block7 failed the unchanged combined atol=rtol=2e-5 gate. These are128-target-row
arithmetic checks with real weights and cast captured prefix, not full-sequence
FP32 or FP32-generated prefill.

| Block7 metric | Rank0 | Rank1 |
| --- | ---: | ---: |
| Elements | 262144 | 262144 |
| Violations | 3 | 0 |
| Nonfinite | 0 | 0 |
| Maximum normalized tolerance ratio | 1.265401175 | 0.477087442 |
| Maximum absolute error | 0.000198364258 | 0.0000762939453 |
| RMSE | 2.39869135e-6 | 1.59968776e-6 |
| Relative L2 | 3.51008847e-7 | 2.33235819e-7 |
| Reference at largest absolute error | 229.699005127 | 191.764297485 |
| Actual at largest absolute error | 229.698806763 | 191.764373779 |

Across both ranks,3/524288 elements violated this block's gate. A small aggregate
relative error does not override those violations. The location with the largest
absolute error is not necessarily the location with the largest normalized ratio;
the former is reported above. There is no one-ULP claim. The cause has not yet
been localized; bit-exact exchange does not establish equivalence of differently
shaped projection/attention/MLP arithmetic.

Both ranks recorded the same failed gate and exited. Full-length BF16 independent
comparisons,1/4/32-block chains, final projection parity and timed repeats were
**not reached**. There is no real-weight Ulysses-versus-single-GPU latency result,
no full-image equivalence and no justification to proceed to40-step trajectories.
Do not substitute the earlier synthetic communication times for these missing
measurements or widen tolerances to bypass the failure.

## Observed resources

| Stage | GPU0 physical sampled peak | GPU1 physical sampled peak | Sampled container memory peak |
| --- | ---: | ---: | ---: |
| Capture | 17280 MiB | 604 MiB | 43,757,379,584 bytes |
| Replay through failed block7 | 1426 MiB | 2022 MiB | 2,923,368,448 bytes |

Capture's Torch peak allocated/reserved were17,588,168,192/17,779,654,656 bytes.
Replay stopped before benchmark peak reports; the table therefore gives physical
samples, not instantaneous allocator maxima. Charged/shared page-cache attribution
can differ between successive containers; sampled cgroup memory is not aggregate
host residency of the retained capture. Host available-memory guards also ran.

Sampled GPU utilization mean/max: capture GPU0 7.99%/100%, GPU1 9.97%/16%; replay
GPU0 2.59%/27%, GPU1 9.03%/31%. Stage averages include load, disk/hashing and
synchronization; GPU1 also includes desktop work. They are not isolated kernel
utilization or evidence of speedup. Peak CPU66.5 C, GPU0/1 71/65 C. Machine-readable
power/memory-controller utilization and phase details are in result.json; complete
samples/logs remain in task-4/real-weight-reviewed-9304851.

The next diagnostic should reuse the preserved block7 capture and compare
intermediate normalized activations, Q/K/V, attention output, projection and MLP
on identical inputs, separating layout/distribution from kernel-shape rounding.
Any changed GPU diagnostic needs the same review/CI/maintenance gate. Production
API and numerical tolerances remain unchanged.
