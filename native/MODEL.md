# Checkpoint-backed text model

`kadan::cuda::Model` is the configurable successor to the reviewed four-layer
synthetic `Stack`. It executes the same original decoder producers over every
validated text layer, with one reservation, device arena and publication cursor.
The old synthetic API remains available for its existing regression tests.

This milestone implements an executable single-device token-ID → BF16 logits /
greedy-token path. Actual Qwen checkpoint execution has **not** been validated.
CPU tests read miniature fixture payloads; installed-model inspection reads only
bounded metadata. CUDA compilation does not establish GPU numerical correctness.

## Contract

- Text only, batch one, sequential input IDs, 1–256 validated layers. Kernel
  dimensions impose the existing tighter limits; the installed 40-layer model
  fits them. Vision and MTP remain excluded.
- All routed and shared experts remain packed and resident. No host expert cache,
  cross-device transfer, batching, prefill optimization or production worker IPC.
- Canonical FP8/NVFP4 decode performs the scale products in FP32, rounds each
  decoded weight to BF16 (nearest-even), then multiplies BF16 activations with
  FP32 GPU accumulation. Outputs round at the existing layer boundaries.
- This explicitly matches the **weight-only** semantics of
  `api/inference/llm/qwen.py::project` and `quantization.py::nvfp4_linear`.
  Calibration input scales are validated but not applied. This is not a claim of
  NVIDIA activation-quantized FP8 execution. CPU reference accumulation and GPU
  reductions can differ; actual-model logit tolerances still need review.
- New BF16 launchers preserve the FP32-decoded contract of existing projection
  entry points. Decoder and MoE owners can select the explicit BF16 mode. The
  older standalone attention owners reject it; checkpoint execution uses the
  model's borrowed decoder producers.
- `generation_config.json` supplies 1–16 distinct EOS IDs. The installed IDs are
  **248046 and 248044**. Sampling parameters are not used: this boundary explicitly
  selects greedy decoding with lowest-ID ties. Tokenization, templates and the
  production sampling policy remain outside native.
- Text RoPE uses scalar sequence positions, equivalent to equal text positions
  across the three MRoPE sections. No multimodal position handling is implied.

## Binding, loading and ownership

`model::Layout` consumes a validated `ModelManifest`, derives each decoder plan
and binds every text item exactly once to its final arena destination. It has no
CUDA dependency and reads no payloads. Its immutable public views borrow the
manifest; both lifetimes belong to the model owner. Unknown/missing roles,
unsupported shapes, excess context and invalid EOS sets fail before execution.

The owner first reserves its host envelope, then constructs metadata and layout.
`Resources::resize_loading` atomically extends that same ticket for the complete
arena plus explicit device headroom. A failed extension preserves the original
reservation. Allocation also checks current physical CUDA free memory and SM86.
No payload is read until admission and arena allocation succeed.

Dense BF16 tensors stream through bounded staging chunks. FP8 and NVFP4 tensors
stream as complete packed row tiles, directly into their final regions. No dense
expert bank, full host checkpoint image or per-layer device owner is created.
Staging must hold the largest packed row plus its scalar: **4,100 bytes** for the
installed model. The default is 1 MiB. Finite BF16 dense values, canonical scales,
calibration values and BF16 representability are checked before each upload.
Magnitude bounds validate quantized weights without allocating dense scratch.

Pinned shard descriptors are checked throughout and after loading. The model
becomes resident only after every binding, RoPE frequency and zeroed state has
been uploaded and synchronized. Partial loads never become executable. Failed
cleanup retains the reservation and raises quarantine; no automatic retry occurs.

Every token uses the one model cursor. A layer cannot publish progress. Selection
returns only after all layers, normalization, head projection, selection and host
copy succeed. Cancellation/numerical failures invalidate all state until a
physical reset; uncertain CUDA recovery quarantines the arena. Runtime errors
poison the owner. Capacity/EOS rejection launches nothing. Successful close frees
model allocations and releases the ticket; failed close retains charges and
never retries automatically. `read_logits` is diagnostic: discard its entire
caller-owned destination if it throws.

The resource ledger is still local to this native owner. A caller must supply a
coordinated memory envelope; creating this ledger does not negotiate with the
running Python service or migrate other workloads. No production Python/API/UI
code or deployed service is changed by this milestone.

## Installed metadata and memory

Metadata-only inspection of snapshot
`1355db6a052410cfd62085d94b58866fd0f2c3c5` confirmed 40 layers, 31,333 executable
bindings, vocabulary 248,320 and both EOS IDs. The new layout retained 40,898,511
requested metadata bytes within a 256 MiB parser/metadata quota.

| Context capacity | Device arena bytes | GiB |
| --- | ---: | ---: |
| 1 | 20,897,995,392 | 19.463 |
| 128 | 20,900,677,632 | 19.465 |
| 2,048 | 20,941,228,032 | 19.503 |
| 32,768 | 21,590,034,432 | 20.107 |
| 262,144 | 26,434,455,552 | 24.619 |

Capacity 128 was verified by the new metadata-only binary against the installed
checkpoint. Other rows are arithmetic from the same existing per-layer plans;
they are not measured allocations. State/workspace are included in the arena.
CUDA runtime/driver overhead is covered by separate explicit headroom, not by a
claim that requested bytes equal total device usage.

Defaults reserve **258 MiB host**: 256 MiB bounded metadata/producer storage,
1 MiB staging and 1 MiB fixed control/fd/allocator headroom. Requested allocator
bytes are bounded, not a process RSS cap. The harness separately reserves its
993,280-byte logits buffer. OS page cache and unrelated processes are outside
this ledger. Device admission reserves arena plus 512 MiB headroom by default.

At capacity one this requires **21,434,866,304 free/admitted device bytes** with
that headroom. The observed running-service reservations cannot accommodate it;
even that service's GPU-0 capacity (20,014,117,683 bytes) is below the requested
native envelope. An explicitly coordinated test window/envelope is required.
Do not change running-service limits or unload residency to make a test pass.

## Tools and artifact contract

Safe metadata-only planning (CPU build):

```
kadan-model-plan ROOT CAPACITY METADATA_BYTES
```

The CUDA correctness capture harness requires every execution argument:

```
kadan-model-correctness --execute ROOT DEVICE CAPACITY HOST_BYTES DEVICE_BYTES HEADROOM_BYTES TIMEOUT_SECONDS OUTPUT_FILE TOKEN_ID [TOKEN_ID ...]
```

It accepts 1–8 explicit input IDs, capacity 1–128 and a 1–900-second fail-stop
watchdog. Metadata preflight validates IDs and envelopes before CUDA access.
A stalled runtime exits 124; errors stop the process without another request or
retry. Prompt inputs ignore predicted EOS termination while reporting EOS in each
record; there is no hidden generation loop. Output is created with `O_EXCL` and
`O_NOFOLLOW`; existing artifacts are never overwritten. No timing or throughput
measurement is reported.

An artifact is valid only with process exit zero and its complete `DONE` marker:
all values are little endian, header `uint32[4]` = magic `0x4b4d4331`, version 1,
vocabulary, record count; each record is `uint32[4]` = input ID, selected ID, EOS
boolean, committed-input count, followed by `vocabulary` FP32 logits; the final
`uint32` marker is `0x444f4e45`. Discard files from failed/timed-out runs.

CPU comparison against an independently produced reference capture:

```
kadan-model-compare EXPECTED ACTUAL ABS_TOL REL_TOL
```

The comparator bounds sizes, requires complete files, finite logits, matching
input/selected/EOS/progress records and internally correct greedy selection. It
reports maximum absolute error and rejects values outside
`ABS_TOL + REL_TOL * abs(expected)`. Set tolerances during independent review,
not after looking at failures. A successful native capture alone is not evidence
of actual-model parity.

## Verification and next authorized experiment

CPU fixtures serialize canonical safetensors/config/index/generation files for
4, 8 and 40 layers, plus the representative four-layer shape below. Independent
Python prefix equations use non-power-of-two
scales; native CPU references match every layer output and logit exactly, BF16
state exactly, and FP32 recurrence within absolute 2e-5, including reset replay.
The fixture helper tests are new test-only Python and require review with this PR.

Tests also cover chunk-size invariance, all role pointers, insufficient envelopes,
invalid calibration and dense/quantized payloads, generation errors, shard mutation,
mid-load cancellation, both EOS IDs, no per-step allocation, one ticket when the
other 1,023 ledger slots are occupied, late state mutation failures, final-copy
failure, quarantine and retained reservations after cleanup failure. A CPU fake
runtime exercises the actual standalone capture program, overwrite prevention,
artifact validation/comparison and the watchdog's exit-124 path.

Before actual-model work, separately authorize the tiny checkpoint GPU harness
and compare its capture with the fixture generator's `reference.capture`. The new
BF16 kernel mode and streamed owner have only CPU/fake-runtime verification here.

After independent review and explicit authorization of the memory window and
actual payload reads, the smallest proposed experiment is:

1. Produce a complete reference capture for one explicit text BOS ID **248044**
   using the existing weight-only reference, with pinned snapshot/config and
   numerical settings. Run it separately from the native load. Review absolute
   and relative tolerances before comparison.
2. Verify exact native commit/binary SHA, idle service state, physical free memory
   and coordinated ledger envelope. Use one GPU, context capacity **1**, one input
   ID, a reviewed load/execution timeout and a fresh output path. The static path
   contains **2,314 kernel launches** for that one input, with top-8 routing; no
   autoregressive loop or benchmark is included.
3. Compare the complete logits and selected token, verify zero final reservations,
   and inspect device residency after process exit. Stop on any mismatch/error.
   A second input, replay, another device or performance test requires a separately
   agreed next stage. None of these actual-checkpoint GPU stages has run here.

## Review-gated tiny-checkpoint plan

The review fixtures now use distinct FP8 and NVFP4 scales at every layer, including
layers separated by four positions. A fourth fixture has four layers, hidden
width 256, linear key/value head counts 2/4 with dimensions 4/4, full head dimension
8 with rotary dimension 4 (frequencies 1 and 0.01), and 12 experts with top-8
routing. Its 256-column FP8 and NVFP4 projections exercise all four reduction
warps. It remains a bounded synthetic fixture, not the installed model's shape
or a performance test. `tests/model_equations.py` supplies independent prefix
matrix equations for this shape and requires separate review with the PR.

CPU comparisons explicitly reject nonfinite expected and actual values. Capture
comparator tests reject NaN/Inf, incorrect greedy/tie selection, EOS/progress
mismatches, truncation/trailing bytes and malformed bounds. They test absolute,
relative and combined tolerance boundaries and the adjacent float outside each
boundary. Fake-runtime self-capture comparisons establish **lifecycle and format
only**, never numerical parity. `reference.capture`, generated by the independent
Python equations, is the proposed GPU oracle.

The complete SHA256 pins for generator sources, all fixture files, independent
reference captures and the compiled harness/comparator are in
[`MODEL-GPU-PLAN.json`](MODEL-GPU-PLAN.json). The local fixture directory recorded
there contains only synthetic files. Regeneration must reproduce every hash;
any source/binary/fixture change requires re-review. These pins supersede earlier
fixture hashes and single-input proposals. Capacity is **8** and input IDs are
**2, 7, 11**, with three records in each 260-byte capture. Capacity is not encoded
in the capture header; pin it in the command and this plan. Record input IDs,
progress and EOS flags are also pinned in the JSON.

All GPU execution remains held. After independent review, each stage needs its
own explicit authorization; no automatic stage advancement or retry. Each uses
one model allocation/free pair, one model reservation, one 64-byte host logits
reservation, a 270,532,672-byte host envelope and 536,870,912 bytes of device
headroom. Arena/envelope and static forward-launch counts are recorded per stage
in the JSON. Launch counts exclude driver-internal memcpy/memset work. Default
metadata and staging quotas are 256 MiB and 1 MiB; allocator/context overhead,
OS page cache and process RSS are not synonymous with requested allocation bytes.

Each authorized command would use a 30-second internal watchdog plus an external
45-second timeout with a five-second kill grace (maximum 50 seconds including
preflight). Use GPU 0 only after verifying its identity, idle service queue,
physical free memory at least the pinned device envelope, and acceptable
power/temperature under the unchanged existing settings. Capture before/after
telemetry. A fresh output path must be outside the pinned fixture directory.
Require exit zero, DONE, exact three records and zero final reservations. Compare
against **reference.capture with absolute=0 and relative=0**. Stop at any mismatch,
timeout, failure, changed hash or unexpected residency; do not widen tolerances,
rerun, reset the GPU, unload models or change a service to get a pass. This prompt
path reports EOS but ignores predicted EOS termination, and does not test GPU
reset or terminal-EOS behavior.

The harness currently reopens ROOT between metadata preflight and loading and
only compares vocabulary/arena sizes. A same-shaped replacement is not detected
as an identity mismatch. Stat/timestamp checks during loading do not authenticate
a snapshot cryptographically. For the proposed experiment, externally verify all
pinned hashes immediately before and after execution, protect the fixture from
writers throughout, and invalidate the result on any change. An in-process
manifest identity contract remains a follow-up; this experiment does not claim
to close that general TOCTOU gap. Actual-model payload reads, unloading and
service changes remain separately held.
