# Independent full-trajectory harness — source review required

This is a separate review-only harness implementing FULL_TRAJECTORY_PROTOCOL_V1.md,
including the review amendments: eager repeatability (all prediction/latent/image
gates) must pass before any candidate run; raw VAE output must be finite before
normalization/clipping and pinned uint8 rounding. No full trajectory has executed.
The live API and production Ulysses source are unchanged.

## Execution and independence

`launch_trajectory.py` runs one case per separately reviewed window: `reference`,
then `repeat`, then `candidate`. It verifies the previous complete artifact manifest,
hashes, settings, criteria, exact harness commit and successful API restoration.
Candidate admission requires a passed repeat manifest bound to this exact reference.
A failed or incomplete repeat cannot unlock candidate execution. Each case requires
an independent protocol/source review record containing frozen SETTINGS and CRITERIA,
the case, exact commit and `approved-for-bounded-execution` decision, plus green CI.

Each worker constructs its own full pinned pipeline, CPU generator seeded 42,
scheduler, conditioning, initial latents and prefix caches. It uses 2048², 40 steps,
BF16, component CPU offload, VAE tiling, no compilation and unchanged defaults.
All visible checkpoint files are hashed and compared across runs, alongside pipeline,
transformer, scheduler and VAE source hashes, configuration and input conditioning.
Initial latents must match exactly; no reference values are supplied to generation.

The candidate uses two complete pipeline replicas. Both perform independent eager
prefill. `trajectory_adapter.py` shards target hidden rows only at the first cached
block, keeps local output through all 32 blocks, shards the final norm mask, then
gathers projected rows once before each rank's own scheduler update. Original eager
and repeat block implementations remain untouched. Prefix head shards are built from
each rank's own eager prefill and compacted once. Candidate ranks check exact agreement
on initial inputs and their gathered predictions/scheduler latents. Both independently
decode and compare images. This duplicated preparation/decode is deliberate in the
correctness harness and must be included in its resource and timing interpretation.

`run_trajectory.py` wraps scheduler.step to snapshot the actual consumed prediction
and returned latent, then returns that same path's output object. References are
loaded only for CPU comparison and never returned to the scheduler/transformer.
The StepOrder contract requires exactly one extract step followed by 39 cached steps,
each paired with its own scheduler update. Both ranks participate in failure consensus;
only rank 0 writes the regular trajectory artifacts. The first failed comparison can
save actual/reference tensors per rank in a separate fixed reserve.

## Fixed image comparison point

The raw tensor returned by VAE decode is checked for nonfinite values before calling
the pinned image processor, so clipping cannot hide infinities. The processor's NumPy
output is float32 NHWC `(1,2048,2048,3)` in [0,1], display RGB interpreted as sRGB,
without additional gamma/profile conversion; comparisons are not linear-light metrics.
Use the pinned `numpy_to_pil` rounding to produce RGB uint8 `(2048,2048,3)` and PNG.

Predictions/latents retain atol=rtol=0.02, zero violations/nonfinite. Float RGB retains
max absolute 2/255 and mean absolute 0.5/255. Uint8 RGB retains max channel difference
2 and mean absolute channel difference 0.5. Both pixel limits must pass. The eager
repeat must satisfy the same gates before candidate output is observed. Thresholds
are not selected from output data.

## Streaming storage plan

Only initial latent, 40 predictions, 40 post-scheduler latents, final normalized RGB,
uint8 pixels, one PNG and compact identity/reports are stored. No block activations
or weight packets are recorded. Tensor snapshots clone compact CPU storage so a view
cannot accidentally serialize its backing allocation. The writer rejects overwrites,
checks projected bytes before writes and actual file bytes after serialization; every
artifact has a size and SHA. Host monitoring counts all historical and new artifacts.

| Budget item | Bytes |
|---|---:|
| Retained historical artifacts measured at preparation | 31,748,605,262 |
| BF16 step snapshots per run (81 × 16384 × 64 × 2) | 169,869,312 |
| Float RGB per run | 50,331,648 |
| Uint8 RGB per run | 12,582,912 |
| PNG cap per run | 16,777,216 |
| Per-run serialization/metadata margin | 5,439,488 |
| Per-run cap | 255,000,576 |
| Three run caps | 765,001,728 |
| Failure reserve, both ranks | 268,435,456 |
| Logs/telemetry reserve | 134,217,728 |
| Projected total | 32,916,260,174 |
| Unchanged total ceiling | 34,359,738,368 |

Admission recalculates actual bytes. For later cases it conservatively counts already
written prior-case artifacts plus all three run caps again. Inputs and old evidence
are read-only and never deleted by this harness. A failure keeps its partial outputs.

## Timing, guards and recovery

The request clock starts before pipeline invocation and ends after its complete output,
CUDA synchronization and audits. It includes conditioning/text encoding, offload,
prefill, all denoising/scheduler steps, VAE, image conversion/writing and numerical
checks. Main-entry wall time additionally includes hashing/model load; supervisor and
launcher timings include process/container startup and API restoration respectively.
Candidate reports per-rank and slower-rank request time. There is no subtraction of
auditing overhead and no speedup calculation. Audit work differs by case; these are
correctness-run wall times, not an end-to-end production performance benchmark.

Reuse the existing image. CPU pools remain one intra-op and one inter-op thread per
process, aggregate two-CPU quota, with cgroup throttling telemetry. Five readings below
60°C precede any pause; runtime CPU must remain below 80°C, GPU below 90°C. Thermal
records identify every sensor and the latest timestamped rank phase before rejection.
The existing lock inode, 128-GiB RAM/no swap, 22-GiB per-GPU allocator, 32-GiB artifacts,
120-second collective timeout, 900-second stage cap, 2,100-second pause ceiling and
600-second recovery reserve are unchanged. No timeout extension is assumed for a full
image. Rank cleanup uncertainty quarantines GPU ownership; otherwise restore and verify
the exact captured API container/config/mounts and native model readiness.

## Validation and remaining review

CPU tests cover independent prediction/scheduler ordering, incomplete/restarted
trajectory rejection, predecessor artifact integrity, pixel maxima/means and unsigned
subtraction, float/nonfinite gates before clipping, compact snapshots/no overwrite,
storage limits, shard-once/local-state/gather ownership, read-only default launch and
owned-process cancellation cleanup. GPU numerical correctness, complete-run memory
fit and time fit remain unverified. Exact-source independent review and execution
authorization are required before each bounded run. No API merge or activation.

Validation at publication: **61 CPU tests passed** in the pinned image with no GPU
access. Python syntax checks, whitespace checks and the launcher's read-only plan
also passed. All three execution cases must use the same reviewed frozen source
commit; keep run evidence outside the checkout until the sequence completes so
publishing interim results does not change its identity.
