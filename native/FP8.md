# Original SM86 FP8 projection

This milestone adds fused E4M3FN weight decoding and FP32 matrix-vector execution
for the mixed FP8 projections accepted by the native checkpoint reader. It uses
original CUDA scalar code, with no external inference engine or kernel code.
The worker, Python adapters and API/UI remain unchanged. Other modalities are
not migrated. This is a primitive, not a full-model engine or performance result.

The inspected Qwen checkpoint has FP8 `in_proj_qkv` weights with a scalar F32
scale; its `lm_head` is NVFP4, as recorded in [CHECKPOINT.md](CHECKPOINT.md).
A future model executor must dispatch each tensor using its validated encoding.
FP8 is not an assumption about the eventual output head.

## Admission and execution

`plan_fp8` accepts canonical row-major E4M3FN bytes and one positive finite FP32
multiplier or one multiplier per row. Any positive row/column size up to INT_MAX
is supported; columns need not be aligned or even. Extra block scales, shape
mismatches, both NaN encodings, invalid multipliers and exact decoded weights
outside FP32 range are rejected on the CPU before CUDA allocation or admission.
Signed zero, subnormals, underflow to zero and exactly +/-FLT_MAX remain valid.
The exact weight-range test uses a double product before FP32 rounding, including
values that exceed FLT_MAX but would round back to finite FLT_MAX.

Planning first uses the canonical format validator, then scans decoded-weight
range without dense allocation. These admission scans are not repeated per token.
A returned `Fp8Plan` describes allocation only; it is not a validation token.
`Fp8Projection` always validates its own constructor input. The caller must keep
host spans alive and immutable throughout construction. The constructor uploads
weights and scales into private device storage and synchronizes before returning;
then the caller can release or change the original buffers. No host weight spans
are retained and no public interface exposes device storage for mutation.

One 256-byte-aligned slab holds FP8 weights, FP32 scale(s), FP32 input/output and
status, including padding in the explicit budget. The shared `Resources` ledger
reserves the slab before allocation and releases it only after acknowledged
cleanup. Host buffers and CUDA context/module/driver overhead need separate
admission and headroom. This requested-allocation accounting does not measure
physical VRAM or coordinate with a separate Python ledger by itself.

The implementation factors #101's owner into a private template shared by
`Nvfp4Projection` and `Fp8Projection`. Encoding-specific operations only select
planning, scale upload and the kernel. Device/thread ownership, pinning, legacy
stream ordering, poisoning, cleanup and uncertain-reservation retention follow
[CUDA.md](CUDA.md). The existing NVFP4 numerical kernel is unchanged.

FP8 uses one 128-thread block per output row. Lanes decode byte weights in
registers, multiply by the admitted scalar/row scale with explicit FP32
round-to-nearest, accumulate FP32 FMA and reduce four warps. There is no dense
weight scratch or per-call allocation. Because the immutable weights were
validated at admission, the kernel does not recheck every weight for NaN or
pre-round range on each token. It still detects nonfinite reductions. Each call
validates input dimensions/finite values, pins storage, copies input, launches
once, synchronizes, checks status, copies output on success, and unpins.

The CPU reference rounds decoded weights to FP32 and accumulates in double;
GPU accumulation is FP32 and has a different reduction order. FP32 reductions
may overflow where a double sum would not. Numerical errors remain errors;
no output is returned as a fabricated successful result. Runtime failure and
uncertain cleanup still require a fatal-worker policy in a future supervisor.

## Verification and one proposed GPU validation batch

Seven CPU CTest cases pass with ASan, LSan and nonrecovering UBSan. The new test
exhaustively compares the shared kernel decoder against the CPU reference for
all 254 finite E4M3FN encodings, including signed zero. It covers scalar/row scale
layout, odd/tail dimensions, budget rejection, NaN/invalid scales, exact maximum,
subnormal/underflow and both signs of the pre-round overflow boundary.

Both kernels, their shared owner and both opt-in parity executables compile and
link for fixed SM86 using the installed CUDA 12.0/GCC 12 toolchain. Compile
parallelism is two. CTest contains only CPU tests. Neither GPU executable has
been run for this revision. #101's earlier six-case NVFP4 smoke pass does not
validate the shared-owner refactor or the new FP8 kernel.

After independent review and a **new explicit authorization**, propose one batch
of **twelve synthetic kernel launches on one RTX 3090**, executed sequentially:

| Executable | Launches | Coverage |
| --- | ---: | --- |
| `kadan-cuda-parity` | 6 | Existing NVFP4 four ordinary shapes, exact subnormal, expected pre-round overflow rejection; exercises the shared-owner refactor. |
| `kadan-fp8-parity` | 6 | Scalar 1x1, per-row 3x17, per-row 17x1025 twice with reversed input and the same persistent storage, exact subnormal 1x16, expected reduction overflow 1x2. |

The FP8 harness computes the CPU reference before replacing source host weights
with NaNs and source scales with NaNs after construction. Subsequent successful
calls must use the immutable uploaded copy, including reuse for the second token.
The finite parity tolerance is `1e-5 + 64 * FLT_EPSILON * sum(abs(weight * input))`;
subnormal comparison is exact. Overflow expects rejection with untouched caller
output. Every fixture closes and checks reservation release.

Maximum requested live operator slab is 46,848 bytes (existing NVFP4); FP8 peaks
at 22,784 bytes. Each harness enforces a 64 KiB operator budget. Fixtures remain
below 1 MiB host data. CUDA context/module overhead is additional and unmeasured.
No model files, full-model loading/generation, timing loops, throughput benchmarks,
second-device work, power changes or deployment are part of this proposed batch.
The prior six-launch approval is consumed and does not authorize this batch.

Before any authorized run, verify the final reviewed head and binary, select one
GPU, and check read-only request/queue activity and device utilization. Do not
stop services to make room. Stop on conflict, runtime errors or instability; do
not rerun a failed harness without a separately scoped next step. Post-run read-only
samples and ledger release checks are bounded correctness/cleanup observations,
not evidence of electrical safety or a throughput claim.

Compile-only commands (do not run the binaries without that new authorization):

```sh
cmake -S native -B /tmp/kadan-fp8-cpu -DKADAN_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all"
cmake --build /tmp/kadan-fp8-cpu --parallel 2
ctest --test-dir /tmp/kadan-fp8-cpu --output-on-failure
cmake -S native -B /tmp/kadan-fp8-cuda -DKADAN_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/kadan-fp8-cuda --parallel 2
```

## Remaining engine milestones and integration review points

1. **Full checkpoint manifest and placement.** Validate model config, shard index,
   tensor-role shapes and encoding dispatch; admit bounded host loading buffers,
   persistent weights and transfer scratch. The reader currently handles selected
   rows from an explicitly selected shard, not a full model. Connect its host
   budget to the global resource envelope without double counting.
2. **Device-resident layer primitives.** Add original normalization, residual,
   activation, positional, embedding and attention/recurrent-state operations,
   plus BF16 rounding semantics where required by the model. Add device-buffer
   interfaces so layer chaining avoids this development API's per-projection
   host copies and synchronizations. Validate tiny layers against CPU references.
3. **MoE and sequence state.** Implement routing, shared/routed expert execution,
   expert residency/eviction, prefill/decode state and cancellation-safe ownership.
   Dual-device placement and transfer/synchronization must account for every live
   host/device copy and temporary before changing admission. No full-model fit
   or performance is established by the primitive slabs.
4. **Native model loop and output path.** Assemble layers, encoding-aware output
   projection, logits, sampling and stopping into native decode; retain Python
   tokenization initially if useful. Validate deterministic tiny end-to-end cases
   before separately authorized real-checkpoint/model runs and benchmarks.
5. **Opt-in process integration, requiring Andrew's review of production Python
   changes.** Extend `native/src/worker.cpp` beyond accounting-only commands with
   bounded requests/events, IDs, cancellation, shutdown and crash handling. Add a
   supervisor used by `api/services/runtime.py` and the LLM feature/adapter in
   `api/inference/llm/feature.py` and `qwen.py`, with streaming/conversation
   compatibility in `generation.py`, `streaming.py` and `conversation.py`.
   Coordinate envelopes with `api/memory_manager/__init__.py`, `queue.py` and the
   shared resource manager: Python can remain the global admission authority,
   and must retain the worker envelope until cleanup or termination is confirmed.
   Existing video/audio/Decisions workloads keep their adapters; a later reviewed
   migration can extend native scheduling without creating two budget authorities.

These are follow-on milestones, not implementation claims for this PR. Keep the
first production adapter opt-in and preserve API/UI behavior. Optimize only after
correctness and separately authorized measurements; the dual-3090 50+ decode
tok/sec objective remains unmeasured.
