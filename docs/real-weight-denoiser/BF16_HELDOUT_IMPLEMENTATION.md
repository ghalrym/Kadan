# BF16 held-out timestep extension — source review required

This extension implements the already declared zero-based steps 20 and 39 from
BF16_APPLICATION_PROTOCOL_V1.md. Neither held-out case has been executed. The
first-slice result at 67b3768 remains passed; the FP32 protocols remain failed and
controlled-accumulation candidate v1 remains rejected. Production is unchanged.

## Capture and replay

Run one step per maintenance window, step 20 first. The pinned unchanged eager
pipeline advances normally from seed 42, the original red-apple prompt, 2048×2048,
40 configured scheduler steps, production scheduler/default conditioning, BF16,
component CPU offload and no compilation. The hook counts prefill as step 0 and
records every actual timestep through the selected cached step. Capture stops at
that step's final projection, before scheduler update and VAE decode.

`capture_holdout.py` writes only activations, modulation, rotary values, prefix
cache, mask and expected outputs. Each block and final norm/projection verifies
its live weights against the original immutable packet; no new weight packets
are serialized. `holdout_contracts.py` binds each context to that packet SHA and
the original manifest. `replay_holdout.py` references the original mmap-backed
weights and overlays the selected context. The existing first-slice implementation
is unchanged; the extension uses the same unmodified production Ulysses source.

Acceptance remains atol=rtol=0.02, zero violations and zero nonfinite values at all
full-shape block outputs, every block of independently advancing 1/4/32-block
chains, and final norm/projection. Transport ownership and roundtrip checks are
bit-exact. All ranks participate in global audits and coordinated failure. There
is no timing stage, tolerance adjustment, candidate selection or production
activation. Failure evidence is retained and the ladder stops.

## Storage and cleanup

CPU-only shape planning against the verified existing capture measured:

| Item | Bytes |
|---|---:|
| Existing immutable capture | 22,869,681,264 |
| One step's context tensors | 8,876,752,896 |
| Serialization margin | 553,648,128 |
| Evidence reserve | 536,870,912 |
| Projected active total | 32,836,953,200 |
| Fixed limit (32 GiB) | 34,359,738,368 |

The launcher plans before stopping the API. Each write preflights its tensor size
plus margin and checks actual serialized size; host monitoring enforces the total
including prior step evidence. Base packets are mounted read-only. Only exact
owned context files can be removed, after both matching replay verdicts pass and
the exact API has been restored. The original manifest is retained byte-for-byte
and fsynced before deletion; hashes and a deletion receipt remain. Failed,
modified, partial or unowned contexts are preserved. Step 39 admission requires
step 20's two verdicts and receipt to bind the same retained manifest, completed
restoration, and an empty former context directory.

## Execution admission

`launch_holdout.py` defaults to a read-only plan. Execution requires a clean exact
commit, green API/native CI for it, and an independent review record specifying
protocol `bf16-heldout-steps-v1`, source commit, selected step, reviewer, review
reference, and decision `approved-for-bounded-execution`. Coordinate an explicitly
authorized API-only maintenance window after source review. Do not execute merely
because this document or a record template exists.

The unchanged envelope is 2,100 seconds including a 600-second recovery reserve,
at most 900 seconds per stage clipped to the remaining measurement budget,
120-second collective timeout, 128 GiB host RAM/no swap, four CPUs, 22 GiB allocator
per GPU, 32 GiB active artifacts, and existing thermal/free-memory guards. The
existing inference-lock inode is retained. Owned ranks are reaped before releasing
ownership; uncertain cleanup quarantines the GPUs. Restore starts the exact
captured API container and verifies identity, config, mounts, health and native
model readiness. No whole image rebuild is needed.

## Validation and remaining gates

CPU unit tests cover step numbering, shared weight identity, forbidden duplicate
weights, changed weights, pre-write budgeting, path ownership, failed/changed-file
cleanup, both-rank verdict binding, prior-step admission and read-only launch.
The full diagnostic CPU suite passes 38 tests in the pinned image. These tests do
not establish GPU correctness or runtime fit for steps 20/39.

After both reviewed held-out runs pass, prepare a separately reviewed full
independent 40-step trajectory and decoded-image protocol with unchanged settings
and explicit acceptance criteria. Full trajectory, image equivalence, API review
and production activation remain outstanding.
