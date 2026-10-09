# Full independent trajectory and image protocol v1 — proposed, not executed

The three declared BF16 cached-step cases passed. They do not test accumulated
scheduler feedback or decoded images. This protocol proposes the next acceptance
gate before implementation/execution review. No run or production activation is
authorized by this document. Criteria must be independently approved and frozen
before observing a candidate trajectory; no tuning after failures.

## Fixed scope and settings

Use the same pinned image/checkpoint, production BF16 operations, default attention
dispatch, component CPU offload, no compilation, prompt, seed 42, 2048×2048, one
image and 40 configured scheduler steps as BF16_APPLICATION_PROTOCOL_V1.md. Record
the exact scheduler configuration, actual timesteps, guidance/default conditioning,
tokenizer/text encoder, VAE settings, precision flags, model hashes and source SHA.
Any mismatch invalidates a pair. No changed sampler, shortened token shape,
projection accumulation candidate or reduced-step surrogate.

The first case is the unchanged red-apple prompt used by the cached-step protocol.
It establishes acceptance only for that case. Editing/reference-image parity and
broader prompt/seed coverage require separately predeclared cases; do not claim them
from this run. The GPU candidate is the same reviewed Ulysses arithmetic, integrated
into a review-only full-pipeline harness. API and production integration stay separate.

## Independent trajectories

Generate the single-GPU eager reference from scratch and the two-GPU candidate from
scratch in separate owned runs. Use independently initialized generators with the
same seed, independent scheduler instances/history, separate model cache state and
independent VAE decode. Each path performs its own unchanged prefill and builds its
own prefix caches. Compare the initial random latents and conditioning hashes exactly
as an input-identity gate; do not copy intermediate reference activations into the
candidate or reset candidate state between steps.

For each of the 40 actual steps, record timestep and compare the full denoiser
prediction and post-scheduler latent against the corresponding reference. Each path
feeds only its own prediction into its own scheduler and advances its own latents.
Instrumentation must not alter returned tensors, scheduling, defaults or casts.
The candidate gathers target rows in their original order before its own scheduler
update. Validate explicit layout ownership; a mutually cancelling permutation cannot
stand in for a correct layout.

Run and validate the eager repeatability trajectory first, including every prediction
and post-scheduler latent gate, before candidate admission or candidate output observation.

At the end, compare final latent and independently decoded floating RGB output before
quantization. Reject nonfinite raw VAE output before `(image * 0.5 + 0.5).clamp(0,1)`.
The float comparison point is the pinned processor NumPy output: float32 NHWC
`(1,2048,2048,3)`, normalized display RGB interpreted as sRGB, with no added gamma or
color-profile conversion. The comparison is in this encoded RGB space, not linear light.
Use the pinned processor's uint8 rounding through `numpy_to_pil`; no alternate quantizer. Save lossless PNGs of each path and compare the final uint8 pixels.
A successful teacher-forced block comparison cannot replace this trajectory gate.

## Proposed fixed acceptance criteria

These criteria are proposals awaiting independent review, not accepted results:

- Inputs, step count/order, timesteps, settings and tensor shape/dtype: exact match.
- Every denoiser prediction, every post-scheduler latent and final latent: existing
  atol=rtol=0.02, zero violating elements and zero nonfinite elements. Report each
  step separately, worst coordinates and global aggregate; do not average failures away.
- Decoded float RGB in [0,1], before uint8 conversion: absolute tolerance 2/255,
  relative tolerance 0, zero violations/nonfinite; mean absolute error at most 0.5/255.
- Final lossless uint8 RGB: maximum channel difference at most 2 and mean absolute
  channel difference at most 0.5 across the complete 2048×2048 image. Both criteria
  must pass. Pixel comparison is the gate; visual/perceptual measures are diagnostic.
- Same-path repeatability must be checked before attributing a candidate discrepancy
  to parallel execution. If the eager reference fails the same proposed image criteria
  on a repeated reference run, mark the gate inconclusive and investigate; do not
  loosen thresholds automatically or claim candidate equivalence.

Report all raw metrics and paired images even when a later gate fails. Preserve
prior failed FP32 results and the interrupted step-39 attempt. No unexecuted stage
counts as passed. On failure, persist the first failing complete comparison and stop
candidate work under coordinated cleanup; any subsequent diagnostics require review.

## Storage, execution and safety constraints

Reuse the original immutable weights/model cache. The retained active artifact set
already occupies about 31.75 GB of the 32-GiB envelope. Before implementation review,
produce a byte-exact plan for reference predictions/latents, images, logs and failure
evidence, including tensor shapes/dtypes and serialization overhead. Stream only
needed scheduler-boundary tensors; never store every block's trajectory activations.
Do not duplicate weights or delete historical evidence to make room. If the full plan
does not fit, revise the storage protocol for review before running.

Reference, eager-repeatability and candidate runs use separate bounded windows, each
admitted only with idle queue, five CPU readings below 60°C and all existing ownership,
RAM/disk and GPU checks. Keep CPU intra/inter-op pools at one per process, two-CPU
aggregate diagnostic quota, runtime CPU strictly below 80°C, GPU below 90°C,
128-GiB container RAM/no swap, 22-GiB per-GPU allocator, 120-second collective timeout,
900-second stage cap, 2,100-second total pause and 600-second recovery reserve.
Do not extend bounds if a complete stage cannot fit; return for protocol review.

The separate review-only implementation is described in FULL_TRAJECTORY_IMPLEMENTATION.md.
The cached-step harness cannot execute this protocol. First implement and
CPU-test the review-only pipeline adapter, independent-state contracts, tensor storage
plan, cancellation/worker cleanup, labelled thermal telemetry and exact API restoration.
Obtain independent protocol and exact-source review, green exact-head CI and explicit
bounded-execution authorization before any run. Use the existing image/hot-reload
workflow and rebuild only genuinely affected native targets if required.

## Claims and later work

Do not make an end-to-end speedup claim from this correctness run. Any timing study
must predeclare warmup/repeats and include text encoding, offload/transfers, prefill,
all denoising steps, scheduler, VAE and output preparation under identical settings.
Production readiness additionally needs API scheduling/cancellation/memory-accounting
integration and review. Passing this one image case would not authorize activation.
