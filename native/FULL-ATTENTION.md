# Original one-token full-attention sublayer

This milestone connects hidden input to the attention residual for batch-one,
text-only decoding. It does not implement the MoE/MLP, a decoder loop, checkpoint
payload loading, production routing, or multimodal positions. The native CPU and
CUDA implementations are original. No external model/engine/kernel code is copied,
wrapped, ported, vendored or executed. The pinned source supplies mathematical
conventions only. GPU execution remains gated on independent exact-head review.

## Mathematical and dtype contract

Source: Transformers 5.17.0 commit
[`856157a2f3e9594954310df18fdccc31ffddebe9`](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py),
verified source SHA256
`42b6ca1cd9a2f0754c3dec97f0708dfa7e97e1b6edd9fac88f3139e4e36316c2`.
Relevant locations: text rotary frequencies/casts 178–215, split-half partial
rotation 668–710, eager attention 727–750, Q/gate split and forward 754–825,
offset RMS norm 925–939, pre-norm/residual 947–1000.
The existing manifest identifies scalar-scaled FP8 Q/gate, K, V and output
projections, with BF16 input/Q/K norm weights. No tensor payload was read.

Let `B(x)` mean float32-to-BF16 round-to-nearest ties-to-even, rejecting nonfinite
results. Inputs and outputs hold exact BF16 values in float storage. Persistent
K/V uses BF16 bits in `[token,kv_head,coordinate]` order. Weights are immutable
canonical row-major FP8 with scalar FP32 multipliers; auxiliary norms are packed
BF16 on device. Caller-supplied FP32 rotary frequencies are positive and <=1;
there are `rotary_dim/2` of them. For the target model, rotary_dim=64,
head_dim=256, heads=16, kv_heads=2, theta=1e7. Text position is the committed
sequence length, starting at zero, shared across the T/H/W streams. No arbitrary
position IDs, padding, sliding window or spatial MRoPE are supported here.

1. Offset input RMS norm reduces squares in FP32 and returns
   `B((x * inverse_rms) * (1 + input_weight))`.
2. Q/gate, K and V projections each round to BF16. Q/gate rows are arranged
   `[head0 Q, head0 gate, head1 Q, head1 gate, ...]`, not all Q followed by all gates.
3. Q and K each use an offset RMS norm across that head's coordinates, with shared
   learned per-coordinate weights and FP32 reductions; the normalized result
   rounds once to BF16 after multiplication by `(1+weight)`.
4. For each split-half pair within the rotary prefix, angle is FP32
   `position * frequency`. Cosine/sine round to BF16. Each product rounds before
   the BF16 sum: `B(B(a*cos) - B(b*sin))`, `B(B(a*sin) + B(b*cos))`.
   The suffix is unchanged. This follows the eager BF16 tensor operation
   boundaries, unlike the earlier standalone FP32 rotary primitive.
5. Append this token's rotated K and projected V into the bounded cache. Each
   query head maps to `kv_head = head / (heads / kv_heads)`.
6. For all tokens through the current position, compute a FP32 ordered Q·K
   reduction, then `B(B(dot) * (1/sqrt(head_dim)))`. Eager matmul's BF16 output
   and scalar multiplication are separate boundaries. Stable softmax subtracts
   the row maximum and computes exp/sum/division in FP32, then rounds each
   probability to BF16. Unused probability slots are zero and excluded.
7. Weighted V accumulates in FP32 and rounds to BF16. Multiply that output by
   `B(sigmoid(gate))`, then round the product to BF16.
8. FP8 output projection rounds to BF16; adding the original hidden residual
   rounds again to BF16.

All reductions reject nonfinite intermediates; CUDA raises status flags and the
host throws at a blocking boundary. The CPU FP8 oracle accumulates in double;
existing GPU FP8 uses FP32 parallel reduction. Full-attention reductions are FP32
ordered and CUDA uses explicit rounded add/multiply intrinsics. Exact agreement
on the chosen synthetic BF16 goldens is required, but general bitwise equality
with Transformers, quantized libraries, SDPA or FlashAttention is not claimed.
Different fused attention paths can retain extra precision between operations.
The serial-per-head kernels are a bounded correctness foundation, not a throughput
implementation or evidence for the eventual decode-speed target.

## Ownership and admission

`full::Reference` borrows immutable host weights and admits its state/scratch
against an explicit byte budget. It has no per-token allocations. `cuda::FullAttention`
owns four reviewed `Fp8Projection` owners, one full-attention `SequenceState` arena,
and one packed norm/frequency/scratch/status allocation. All are charged to the
shared `Resources` authority. Projection inputs and outputs stay on device.
Host inputs/weights, diagnostics and the shared authority are caller-owned;
construction uses a fixed 2KiB BF16 staging array. No hidden model-sized host copy
or dense dequantized weight matrix is created.

The CUDA owner is creating-thread/current-SM86-device only, uses the legacy stream,
and blocks at status/commit boundaries. Caller device input/output allocations
must remain admitted and pinned; exact alias is allowed, partial overlap is not.
A failed attempted step invalidates the whole sequence until physical reset;
capacity rejection happens before begin and preserves valid committed history.
Reset zeros the complete cache before reopening the sequence. The state cursor's
owner-bound token and commit synchronization remain the authority for progress.
There is no whole-model or multi-device transaction in this class.

CUDA runtime failures poison the owner. `valid()` includes child state health.
Reset, state read, intermediate read and begin errors propagate poison. An
uncertain abort or projection recovery raises `DeviceBufferQuarantine`: caller
buffers must remain pinned/alive through successful explicit close or context
teardown, even after the method throws. Cleanup failure retains reservations and
latches against automatic retries. Keep the shared resource authority alive
independently. Discard output after any error. Diagnostics require valid state;
intermediates additionally require a committed token.

## Independent CPU evidence

The tiny fixture has hidden=3, query heads=4, KV heads=2, head_dim=6,
rotary_dim=4, capacity=4. This exercises two grouped pairs, two rotary frequencies,
an untouched two-coordinate suffix, distinct Q/K/V and per-head gates,
nonuniform offset norm weights, dense output mixing and four signed input tokens.
The fixture's frequencies correspond to theta=100, for visible short-sequence
rotation; they do not stand in for actual-model parameters.

`tests/full_golden.py` is a **new test-only Python generator requiring user review
before merge**. It uses only the standard library; no model/native imports,
network, database or GPU calls. Independent immutable prefix-matrix equations
compute the expected attention outputs, using double dot/softmax arithmetic with
explicit BF16 tensor boundaries. It does not call production helpers or obtain
expected values by running the native implementation. `full_golden.hpp` freezes
all four tokens' complete 48-element K and V caches (including zero unused tail),
16 attention probabilities, 24 core values, 24 gated values and 3 residuals.
The CPU test requires exact equality, then resets and repeats. It additionally
checks capacity rejection, aliasing, invalid input, failed-step rejection,
physical reset and finite Q/K norm weights causing attention overflow after
cache append. All 15 CPU CTest cases pass with ASan/LSan and nonrecovering UBSan.

`full-runtime-tests` compiles the actual owner/projection/state sources against a
CPU-only CUDA shim. It checks reset/read/begin poison, each intermediate-copy
failure, synchronization recovery/quarantine, capacity rejection without enqueue,
allocation/operation bounds and cleanup. No GPU is invoked and the shim makes no
claim about actual driver failure behavior. Regenerate constants with
`python3 native/tests/full_golden.py > /tmp/full_golden.hpp` and compare to the
committed header. The previously approved generators remain unchanged.

## Proposed GPU validation, not yet executed

After independent math/runtime review of the exact published head and explicit
clearance, run each stage as a separately authorized fail-stop process on GPU0:

```
/tmp/kadan-full-cuda/kadan-full-attention-parity --allow-gpu-validation --device 0 --stage N
```

| Stage | Work | Compute launches | Async memsets | Explicit syncs |
| --- | --- | ---: | ---: | ---: |
| 1 | One token, all independent goldens, physical reset, cleanup | 9 | 7 | 21 |
| 2 | Four tokens and reset/replay, capacity rejection, all goldens | 72 | 43 | 78 |
| 3 | Intended Q·K overflow after KV append, invalidate/reject/reset | 7 | 6 | 19 |

All stages use the same tiny shape and **7,424 peak requested device bytes**:
four 1,280-byte FP8 slabs, 512-byte KV arena, 1,536-byte auxiliary/scratch/status
slab and 256-byte caller IO slab. The cap is 64KiB. Seven allocations/free pairs;
no per-token allocation. CPU numeric state/scratch is exactly 952 bytes and host
fixture/reference/diagnostic numeric buffers total below 16KiB. CUDA context,
allocator and driver overhead are additional.

Stage 2's shim asserts 72 launches, 43 memsets, 78 synchronizations, 128 copies,
seven malloc/free pairs and the actual 7,424-byte peak. Setup has 12 H2D copies
(342 bytes), one 512-byte state memset and six syncs. Each successful token has
five status memsets, eight syncs, seven status copies (28 bytes), 12-byte input
and output copies, 192 bytes of KV inspection and 256 bytes of intermediate
inspection. Each reset uses one 512-byte memset, one sync, then two state copies;
close uses six syncs. Stage 1 has 28 total copies; stage 3 has 20. Unexpected
runtime failures can add recovery synchronization before stopping.

Validate exact BF16/cache/probability/core/gated/residual goldens, logical progress,
zero tails, reset-to-zero and zero-ledger cleanup. Stage 3 uses finite BF16 norm
weights near 1e20 so attention logits overflow after append; this intended
numerical error must be distinguished from unexpected CUDA runtime errors.
Unexpected results stop the dedicated process without rerun, GPU reset or
advancement. Preflight includes exact head/binary, idle queue, available budget
and power/temperature baseline; record before/after telemetry. No actual checkpoint
payload loading, generation, benchmark, settings change, deployment or restart.
All GPU harnesses remain absent from CTest. SM86 compile/link passed using
CUDA12/GCC12 with build parallelism two; compilation is not GPU correctness.
