# Complete decoder layer

This milestone composes one batch-one text decoder layer on one SM86 device:

The mixed-stack extension extracts its private borrowed producer into
`decoder_producer.hpp`; standalone behavior and device bounds are unchanged.
The bound cursor reference adds eight metadata bytes on the verified toolchain;
`host_metadata_bytes()` reports the current charge.

```
a = attention_with_input_norm_and_first_residual(x, position)
u = BF16(offset_RMSNorm(a, post_attention_norm))
m = MoE(u)
y = BF16(a + m)
```

Both linear and full attention are supported. Attention already includes the
first residual; it is not added twice. Post-attention normalization uses the
checkpoint's offset scale `(1 + weight)`, FP32 normalization and one BF16 output
boundary. The second residual is BF16. The existing original FP8, NVFP4,
attention and MoE kernels are reused through private buffer views. The two new
kernels implement post-normalization and the final residual. No external engine
implementation is copied, wrapped, ported or executed.

The mathematical reference remains Transformers 5.17.0 at commit
`856157a2f3e9594954310df18fdccc31ffddebe9`,
`src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`. This establishes the
attention/norm/MoE/residual order; it is not a runtime dependency. Eager BF16
boundaries are explicit, without a general fused-backend parity claim.

## Publication and failure

`cuda::Decoder` owns the only sequence cursor. Its private attention and MoE
producers check the active owner-bound `StateStep` and cannot reserve, release,
reset or commit. Their state and scratch pointers never escape to callers.
Existing standalone attention/MoE contracts and implementations are unchanged.

A step validates input/aliasing, pins the one arena, performs attention into
private storage, post-normalization, MoE, and the second residual. It validates
complete state before the output copy. The final output copy, synchronization,
cancellation check and unpin all precede the prevalidated same-thread cursor
commit. No CUDA/resource operation follows publication. There is no allocation
on a successful step or reset; the CPU runtime test counts allocations.

Cancellation is a borrowed `atomic_bool`, checked before work, after attention,
after MoE, after the final residual and after the final copy/synchronization. It
must stay alive during the call. It is cooperative, not kernel preemption. A
cancellation racing after the last check may complete the step normally.

On an attempted-step failure, committed tokens do not advance, diagnostics become
unavailable and the entire participating sequence becomes invalid. Recurrent
state may already have changed; KV may already contain the uncommitted token.
There is no rollback or reusable prefix after failure. A successful physical
reset zeros all state and scratch before exposing a fresh sequence. Numerical,
input and cancellation failures permit reset after quiescence; runtime failures
poison the owner until close.

Caller IO must be admitted/pinned for the duration, with exact alias allowed and
partial overlap rejected. **Discard output after every failure**, including a
failure/cancellation after the final copy. A failed recovery synchronization
raises `DeviceBufferQuarantine`; retain both borrowed spans until successful
close or context teardown. Failed cleanup retains charges and latches against
automatic retry. Diagnostic copies use caller-admitted RAM; runtime failure also
poisons the owner.

## One admission envelope

Metadata planning computes one aligned device arena for FP8 weights/scales,
attention auxiliary weights, post-norm weights, the MoE packed weights and
scratch, persistent attention state, attention scratch, four hidden-sized
intermediates and status. Scratch regions are disjoint initially. Internal
producers have no child allocations/ledger records; expert count does not grow
the resident-record count.

The owner reserves RAM `sizeof(Impl)` and exact arena VRAM **before** allocating
its descriptor object. On the verified toolchain that is 32,360 RAM bytes. One
`cudaMalloc` follows admission. Successful close releases physical resources and
destroys the charged descriptor immediately, even if wrappers remain alive.
Uncertain cleanup retains a conservative reservation. A failed constructor with
uncertain cleanup may destroy host metadata while retaining its reservation;
this overcounts RAM rather than silently undercounting possible live VRAM.

These charges cover owned resident storage. The caller retains the shared ledger,
immutable source fixture/checkpoint views, IO buffers and context overhead.
Upload staging is a fixed 2KiB stack buffer; bounded temporary control values
are not resident heap storage. CPU `decoder::Reference` instead accepts an
explicit **numeric** state/workspace budget (1,292/1,396 bytes for the fixtures),
like the existing CPU oracles; its control objects and borrowed weights are not
claimed to be included in that numeric budget.

This first owner makes weights/state one admission/eviction unit. Shared weights
across sessions and streaming experts need a later residency boundary. The
coordinator is not a multi-layer or multi-device transaction protocol.

## CPU validation

`decoder_golden.py` is a **new test-only Python generator requiring user review**.
It imports only Python stdlib and independently computes explicit matrix
projections, immutable-prefix full attention, expanded-matrix linear recurrence,
MoE, post-norm and residuals. No production/native code supplies expected values.
Existing generators and production Python are unchanged.

The fixture uses hidden 16, four routed experts/top-2, routed width 16, shared
width 32. Linear attention has two key/value heads with dimensions 2, convolution
width 3 and capacity 4. Full attention has two heads/one KV head, head dimension 4,
rotary dimension 2 and capacity 4. Three distinct tokens test state evolution.
Independent frozen attention residual, post-norm, mixture, final output and
physical state are compared, then reset/replayed. BF16 quantities match exactly;
FP32 recurrent state uses absolute tolerance `2e-5` for the independent expanded
matrix evaluation.

The real CUDA coordinator is also compiled against a CPU runtime shim. It tests:

- Late MoE numerical failure after actual fake-state mutation; published count
  stays at the prior token and caller output remains untouched before final copy.
- Cancellation before work, after attention and after final copy; final-copy and
  final-sync failures; uncertain recovery quarantine; launch and status-copy
  failures; current-device-query errors; poisoned reset and cleanup behavior.
- Physically zero state after reset; stale diagnostics and rejected invalid reuse.
- One remaining ledger slot admits the complete owner; insufficient RAM/VRAM
  rejects before allocation; partial-construction cleanup and latched failed close.
- Instrumented actual descriptor allocations while retaining eight closed wrappers
  under a one-owner RAM cap; allocation-free successful step/reset.
- Owner-bound stale/foreign/duplicate/reset capability rejection is also exercised
  using the same `StateCursor` used by private producers. No public mutation view
  or public child step token exists in the coordinator API.

The shim does not emulate numerical CUDA kernels or prove device correctness.
CTest remains CPU-only. All 19 tests pass with ASan/LSan and nonrecovering UBSan.
CUDA12/GCC12 SM86 compile/link is checked with parallelism two.

## Proposed GPU correctness stages — not executed

`kadan-decoder-parity --allow-gpu-validation --device 0 --stage N` is opt-in and
not registered in CTest. Run each stage in a separate fail-stop process only after
exact-head math/runtime review and stage clearance; never automatically rerun or
advance after an unexpected result. Verify head/binary, idle service queues,
GPU baseline/available memory and temperature first; use one GPU and a 30-second
process timeout. No generation, model payload loading or benchmark is involved.

Each stage runs the linear then full fixture **sequentially**, freeing one before
constructing the next. Peak requested device memory is 13,056 bytes for linear
(12,800 arena + 256 IO), or 14,080 for full (13,824 + 256). Peak owner RAM is 32,360
bytes. Caps are 64KiB device / 128KiB host, excluding caller buffers/context. Each
variant uses two allocations and two ledger entries; the full stage totals four
allocation/free pairs, never simultaneously. Cleanup must finish with zero ledger
charges and invalid closed owners. All state must be physically zero after reset.

| Stage | Work | Compute launches, linear + full |
| --- | --- | --- |
| 1 | One token each; all independent intermediates/output/state; reset and cleanup | 28 + 27 = 55 |
| 2 | Three tokens each, reset and replay; six steps per attention type | 168 + 162 = 330 |
| 3 | Deliberate first-expert activation overflow after attention; invalidation, unchanged output sentinel, rejected reuse, reset; pre-cancel/reset | 15 + 14 = 29 |

Per successful step: linear attention uses 10 launches, full attention 9;
post-norm/final residual add 2; top-2 plus shared MoE adds 16. Stage 3 stops after
attention, post-norm, routing, two expert projections and activation. CPU shim
assertions cover successful/replay and late-failure launch counts. Actual GPU
correctness and performance remain unmeasured for this coordinator.

## Next boundary

A small mixed multi-layer stack should move publication outward to the entire
token, retaining the same borrowed producer design. Embedding/final norm/LM head,
checkpoint binding, tokenizer/sampling, shared residency across sessions,
multi-GPU prepare/synchronize/publish, and throughput work remain separate. No
production API/UI/orchestration change or modality migration is part of this PR.
