# Original resident MoE sublayer

This milestone makes the MoE branch executable on caller-supplied, bounded,
resident synthetic weights. Input is already-normalized BF16 in float storage;
output is the BF16 mixture in float storage. Post-attention normalization, the
second decoder residual, global sequence commit, host expert streaming, checkpoint
payload loading and production orchestration are deliberately separate milestones.
No actual-model execution or performance measurement is included.

The CPU and CUDA code is original and reuses our original NVFP4 GEMV kernel.
No external model/engine/kernel implementation is copied, ported, wrapped, vendored
or executed. Mathematical reference: Transformers 5.17.0 commit
[`856157a2f3e9594954310df18fdccc31ffddebe9`](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L847),
experts 847–879, router 882–903, shared expert 906–922. Source SHA256:
`42b6ca1cd9a2f0754c3dec97f0708dfa7e97e1b6edd9fac88f3139e4e36316c2`.
Verified metadata: hidden=2048, 256 experts, top-8, routed/shared widths=512,
SiLU activation. Manifest router/shared-gate weights are BF16; every expert's
separate gate/up/down projections are canonical ModelOpt NVFP4 with 16-column
E4M3FN block scales and an FP32 global multiplier. Input-scale calibration metadata
is not used for activation quantization by this W4A16 path.

## Explicit numerical contract

`B(x)` denotes FP32-to-BF16 round-to-nearest ties-to-even, rejecting nonfinite
conversion. Router/shared-gate weights and inputs must be exact finite BF16 values.
Projection structures, scales and decoded-weight range are validated at admission
without materializing a dense dequantized matrix.

- Router dot products accumulate in ordered FP32, then become BF16 logits.
  Stable softmax subtracts the maximum and performs exp/sum/division in FP32 over
  **all** experts. Select the largest probabilities, with **lower expert ID first
  on exact FP32 probability ties**. Top-k is not silently substituted with a
  logits-only shortcut. Sum the selected probabilities in score order, divide
  each by that sum, then cast each selected weight to BF16. Do not renormalize
  again after that cast. Selection stays in descending score order in diagnostics.
- Execute selected experts in **ascending expert ID**, keeping their original
  routing weights. For each: `g=B(Wgate*x)`, `u=B(Wup*x)`,
  `activation=B(B(SiLU(g))*u)`, `d=B(Wdown*activation)`.
  Accumulate `routed=B(routed+B(d*weight))`, rounding after every expert addition.
- Always evaluate the shared expert with the same gate/up/down boundaries.
  Its scalar gate is `factor=B(sigmoid(B(Wshared_gate*x)))`;
  `shared=B(shared_down*factor)`, then `result=B(routed+shared)`.
  No residual is added by this class.

The pinned top-k backend does not promise tie ordering. The lower-ID rule is a
native deterministic contract, not a claim of backend identity on ties. Ascending
expert-ID accumulation follows the pinned batch-one eager traversal. NVFP4 CPU
projection uses FP32-decoded weights and double dot accumulation; CUDA reuses our
FP32 FMA/warp-reduction kernel. The fixture's exact BF16 goldens must match, but
general bitwise equivalence to Transformers or third-party quantized/fused kernels
is not claimed. The initial routing kernel is serial and dispatch is blocking;
this is a correctness implementation, not throughput evidence.

## One layer arena, honest lifetime and admission

`moe::plan` computes a contiguous packed layout for all routed/shared matrices,
router weights and shared gate, one shared scratch region, selected IDs and status.
`cuda::Moe` uses **one device allocation and one Resources handle per layer**,
independent of expert count. It directly supplies views into that arena to the
original NVFP4 launcher; it does not construct one projection owner per matrix.
This prevents the 1,024-resident ledger limit from being exhausted by expert
projections. A test fills 1,023 entries before admitting the whole fixture layer
and verifies it consumes only the remaining entry.

Successful `close()` destroys the charged descriptor object immediately, even if
the public wrapper is retained. Repeated close is a no-op; `valid()` becomes false
and execution/reset/diagnostics reject closed wrappers. Uncertain cleanup keeps
the object and reservation intact until destruction/context handling.

The same handle charges `sizeof(Impl)` host bytes for the bounded fixed descriptor
and owner table (31,336 bytes with the verified toolchain), plus exact arena VRAM.
The CPU object is allocated before constructor admission; failed admission
immediately destroys that bounded object and performs no device allocation.
Allocator bookkeeping, the caller's Resources authority, CUDA context/runtime and
caller source/diagnostic buffers are additional. The host descriptor table contains
at most 257 experts × three views, with no host model-sized weight copy. Immutable
source weights are needed only through GPU construction, then may be released.
A fixed 2KiB stack buffer packs BF16 router/shared-gate weights during upload.

The CPU reference borrows immutable weights; its explicit workspace budget covers
numeric scratch and selected IDs. Fixed object/container overhead and caller
weights are additional. GPU forward has no allocation. Only the selected IDs
(at most eight unsigned integers) return to host for dispatch; activations remain
on device. Selected IDs are validated and sorted before pointer dispatch. The
arena stays pinned through routing, all experts and the shared expert; eviction
must wait for complete quiescence. Shared/routed intermediates reuse scratch.

Creating-thread/current-SM86-device and blocking legacy-stream contracts apply.
Caller device input/output must be admitted and pinned; exact alias is allowed,
partial overlap is rejected. Numerical/input failure invalidates the workspace
until reset; output must be discarded. Reset zeros the workspace and invalidates
old diagnostics, without reloading immutable weights. Runtime errors poison the
owner until close. A failed recovery synchronization raises `DeviceBufferQuarantine`;
retain borrowed IO until successful close or context teardown. Diagnostics poison
on copy failures. Cleanup failures latch without retry and retain RAM/VRAM charges;
the caller must keep the shared authority alive independently of the failed owner.

This compact arena has its own exact sizing function. Existing manifest placement
still describes projection slabs; no adapter silently treats those estimates as
these physical allocations. Manifest-to-arena binding and any placement update
must be reviewed when implementing actual loading. Host-cached experts and device
slot replacement are not implemented here.

## Independent evidence

Fixture: hidden=16, four routed experts, top-2, routed width=16, shared width=32.
Dense packed NVFP4 weights have different codes, signs, block scales and global
multipliers across matrices/experts. The wider shared branch exercises different
projection dimensions and multiple scale blocks in its down projection.
Four signed inputs change selected expert pairs; a fifth has equal router logits
and tests lower-ID tie-breaking. All five are repeated after reset.

`tests/moe_golden.py` is a **new test-only Python generator requiring separate user
review before merge**. It uses only the standard library, constructs explicit
small decoded matrices from the arithmetic fixture definition, and evaluates
independent dot/softmax equations with BF16 boundaries. It imports no native/model
code, uses no GPU/network/database, and never runs production helpers to derive
expected results. Frozen `moe_golden.hpp` contains selected IDs, logits, FP32
probabilities, normalized selected weights, routed contribution, gated shared
contribution and combined output for all five inputs. BF16 values/IDs compare
exactly; CPU probabilities use absolute 2e-7 tolerance. Regeneration is byte-identical.

All **17 CPU CTest cases** pass with ASan/LSan and nonrecovering UBSan. MoE tests
also cover aliasing, invalid input, rejected reuse, recovery through reset,
activation overflow after routing, parameter/budget rejection and metadata-only
actual-model shape arithmetic. The CPU CUDA shim compiles the real arena owner
and checks admission exhaustion, the 1,024-entry boundary, partial-construction
cleanup, reset/current-device/diagnostic-copy failures, uncertain enqueue,
synchronization recovery/quarantine, and latched cleanup failures with retained
charges. Fake context teardown is test-only; it is not production cleanup retry.
A regression instruments actual C++ owner allocations, retains eight closed
wrappers under a one-owner RAM cap, and checks live owner bytes against the ledger
at each construction/close. It also checks every closed-wrapper entry point and
idempotent close. No actual CUDA runtime behavior or numerical kernels are
emulated by the shim.

CUDA12/GCC12 SM86 compile/link passes with parallelism two. CTest contains CPU
tests only. No GPU stage has executed for this milestone.

## Proposed separately gated GPU stages

Only after independent exact-head math/runtime review and explicit clearance:

```
/tmp/kadan-moe-cuda/kadan-moe-parity --allow-gpu-validation --device 0 --stage N
```

| Stage | Work | Compute launches | Memsets | Explicit syncs | Copies |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | One input, all route/output goldens, reset and cleanup | 16 | 3 | 10 | 49 |
| 2 | Five changing-route/tie inputs, reset/replay, all goldens | 160 | 13 | 74 | 202 |
| 3 | Intended first-expert activation overflow, reject reuse, reset | 4 | 3 | 6 | 38 |

Each process requests **9,728 device bytes**: 9,472-byte owner arena plus 256-byte
caller IO, under a 64KiB cap. Only two allocations/free pairs and two ledger entries
are used. Owner RAM is 31,336 bytes on this toolchain, under a 128KiB host cap;
caller fixture/reference/diagnostic numeric buffers are below 16KiB. CPU numeric
workspace is exactly 692 bytes. CUDA/context overhead is additional.

Arena breakdown: 512 bytes router/shared-gate storage; 6,144 routed expert bytes;
1,536 shared expert bytes; 768 scratch bytes; 256 selected-ID bytes; 256 status
bytes. Setup performs 32 H2D copies (2,752 payload bytes), a 1,280-byte workspace
memset and one sync. Each successful forward launches routing once plus five
kernels per selected/shared expert; performs one status memset, seven explicit
syncs, seven status copies and one selected-ID copy. Harness input/output and
seven diagnostic copies make 17 copies per successful forward. Reset uses one
workspace memset and one sync; close uses one sync. The CPU shim asserts stage 2's
actual peak, resident count, allocations and full operation counts. Stage 3 uses
large finite FP32 global multipliers for routed gate/up matrices so the BF16
activation product overflows after routing and those projections; output must
remain its sentinel because finish was never launched. This intended numerical
failure is separate from an unexpected CUDA failure.

Require exact selected IDs and BF16 logits/weights/routed/shared/final outputs;
GPU FP32 probabilities must be within absolute 2e-6 of independent constants.
Require valid reset, unavailable stale diagnostics and zero-ledger cleanup.
Preflight exact head/binary, idle queue, available budget and power/temperature;
record before/after samples. Stop on any unexpected result without rerun or
advancing stages. No actual-model loading, generation, benchmark, service changes,
power/settings changes or deployment. Parent owns merges; the new Python generator
remains an explicit user-review item.
