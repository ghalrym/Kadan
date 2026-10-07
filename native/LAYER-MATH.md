# Original layer math prerequisites

This milestone adds allocation-free CPU references and original CUDA launch
primitives for RMS normalization, elementwise gates and text partial RoPE. It is
stacked on the native placement/state work. It does not execute an attention
layer, integrate the production service, load a model, or establish throughput.

## Semantic evidence and boundaries

The installed checkpoint declares `qwen3_5_moe`; the [official Transformers
architecture documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe)
confirms that Qwen3.6 uses this model family. The locally installed config and
model reference were read as a mathematical/configuration specification only;
no external kernel or engine implementation is copied, wrapped, vendored or
ported. No model code was imported or executed during that inspection.

Evidence inspected on 2026-10-07:

- Installed config SHA256:
  `58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc`.
  Epsilon `1e-6`, full-attention head dimension 256, linear value dimension 128,
  partial rotary factor 0.25, default theta 10,000,000, interleaved MRoPE sections
  `[11,11,10]`, SiLU hidden activation, attention output gate enabled.
- Installed `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` SHA256:
  `42b6ca1cd9a2f0754c3dec97f0708dfa7e97e1b6edd9fac88f3139e4e36316c2`.
  Semantic sites: rotary 178–215/699–710, gated norm 220–236,
  Q/gate extraction and Q/K norm 791–801, sigmoid output 822,
  shared-expert sigmoid 918, offset RMSNorm 925–939.

These observations distinguish conventions that cannot be inferred from tensor
names alone:

| Role | Math contract |
| --- | --- |
| Decoder input/post-attention/final norm | RMS normalization across hidden channels, multiplier `1 + weight` |
| Full-attention Q/K norm | Same offset multiplier, separately across each 256-channel head, before RoPE |
| Linear-attention output norm | RMS normalization across each 128-channel value head, direct `weight`, then SiLU of its z gate |
| Full-attention output gate | Elementwise sigmoid after attention output, before output projection |
| Shared-expert gate | Sigmoid of scalar shared-expert gate projection; broadcasting is a future caller concern |
| Routed/shared FFN activation | SiLU(gate projection) multiplied by up projection |

Q projection output contains `[Q_head, gate_head]` for **each head**, not one
contiguous all-Q half followed by one all-gate half. Future packing/strided views
must respect that layout. This PR accepts contiguous rows only and does not
implement extraction, Q/K grouping, attention scores, softmax, convolution,
DeltaNet recurrence, residual addition, routing, sampling or worker integration.

## Equations and numerical policy

For width `D`, one row is normalized as

```
r = sqrt(sum(x[j]^2) / D + epsilon)
y[j] = (x[j] / r) * (weight[j] + offset)
```

`NormScale::direct` selects offset 0 and `one_plus` selects 1. The explicit choice
prevents silently applying decoder weights as direct multipliers. Norm rows share
one weight vector. Linear gated normalization is composed as direct RMSNorm
followed by `gate(silu, normalized, z)`; no intermediate host copy is required.

The gate primitive returns `input * sigmoid(z)` or `input * z * sigmoid(z)`.
Its sigmoid uses a sign-dependent bounded exponential so large negative inputs
do not overflow an intermediate `exp`. Gates require one modulation per element;
there is no implicit scalar broadcast.

All public buffers are FP32. CPU normalization accumulates squares in double;
the original CUDA kernel uses a 128-thread block per row, strided partial sums
and a shared-memory double reduction. Gate products and rotary pair arithmetic
also use double intermediates before a checked FP32 result. This is an explicit
correctness-first numerical choice, **not bitwise reproduction of Transformers
FP32 reductions or BF16 execution**, and its performance is unmeasured. A later
BF16 layer adapter must specify intermediate rounding boundaries: in particular,
the installed gated norm rounds normalized values before weight multiplication,
whereas offset RMSNorm rounds after its multiplier. This PR does not silently
claim those BF16 boundaries have been implemented. No `fast_math` option is used.

Text RoPE requires equal scalar positions for all T/H/W streams. The native
frequency constructor prepares `f[i] = float(theta^(-i / (rotary_dim/2)))` once;
caller retains/uploads that vector. The kernel receives device-resident frequencies
and one scalar position, computes `angle = float(position) * f[i]`, and rotates
split-half pairs inside the prefix:

```
a = x[i]; b = x[i + rotary_dim/2]
y[i]                = a*cos(angle) - b*sin(angle)
y[i + rotary_dim/2] = a*sin(angle) + b*cos(angle)
```

Channels after `rotary_dim` pass through unchanged. For this model the prefix is
64 channels, with pairs `(0,32)` through `(31,63)`; the remaining 192 channels of
each head are unchanged. Equal text positions make the spatial recomposition
irrelevant; image/video position streams, MRoPE offsets/deltas and YaRN are
unsupported. Position must be below the explicitly supplied context (maximum
262,144). Valid frequencies are finite, positive and at most one. A caller must
supply the frequencies prepared for the admitted rotary configuration; the kernel
cannot verify a theta from an arbitrary borrowed frequency vector.

## Buffer, resource and error contract

Shapes are bounded to at most 1,024 rows/heads, 65,536 channels and 1,048,576
floats per operation, with exact span lengths. Epsilon must be finite and positive.
Invalid enums, dimensions, positions and spans fail before launch. CPU references
reject nonfinite inputs and out-of-FP32-range results; discard all outputs after
an exception. They allocate no scratch or payload storage.

CUDA functions operate on borrowed current-device buffers and the explicit legacy
default stream. They allocate nothing, never change devices, never copy data to
host and never synchronize. Trusted native callers must:

1. Admit and pin every buffer through the shared resource authority, including
   immutable weights/frequencies, outputs and one separate aligned status word.
2. Ensure pointers are valid on the current SM86 device for their stated spans;
   the primitive validates address overlap/alignment, not allocation provenance.
3. Clear status to zero on the same stream before a chain. Check every launch
   return, synchronize before releasing pins or observing results, then read
   status. Any nonzero flag means the entire chain's outputs must be discarded.
4. Keep inputs immutable until consumption completes; handle CUDA failures with
   the enclosing owner's quarantine/cleanup policy, not an uncharged retry.

Kernels atomically OR bit 1 for nonfinite inputs, invalid frequencies or a result
outside finite FP32 range. The status accumulates across a chain and is never
cleared by a primitive. A failed operation need not initialize every output.
No speculative consumer may use flagged output without discarding its downstream
results. Exact input/output aliasing is supported, including gates aliasing their
modulation and RoPE operating in place. Partial overlaps and output overlap with
normalization weights/rotary frequencies are rejected. Status cannot overlap any
operand. Separate concurrent callers must coordinate their own stream ownership.

## Verification and proposed GPU scope

All eleven CPU CTest cases pass with ASan, LSan and nonrecovering UBSan. New analytic
cases distinguish direct/offset weights, normalization-before-gate, stable gate
extremes, independent split-half numeric goldens, identity position, untouched
suffixes, multiple heads, in-place operation, maximum text position, dimensions,
overlap, nonfinite values and output overflow. An all-FLT_MAX input verifies that
normalization does not overflow the square reduction. SM86 compilation succeeds
with installed CUDA 12.0 / GCC 12, parallelism two. No new GPU execution occurred.

The unexecuted `kadan-layer-math-parity --allow-gpu-validation --device 0` fixture
is not registered with CTest. After independent exact-head review and clearance,
its proposed single run makes **14 tiny launches**, one 33,024-byte allocation,
a 64 KiB requested-device-storage cap and less than 64 KiB host numeric fixtures
(CUDA context/runtime overhead is additional and unmeasured):

- Three norm launches: both scale conventions with two 129-wide rows, and one
  actual-hidden-width 2,048 row.
- Two elementwise gate launches and a two-launch direct norm→SiLU gate chain
  without an intermediate host transfer.
- Four two-head, 256-wide / 64-rotary launches at positions 0, 1 and 262,143,
  plus in-place position 1; CPU parity tolerance `3e-5 * (1 + abs(reference))`.
- Three expected-error launches: nonfinite norm input, gate output overflow and
  zero rotary frequency. Finally synchronization, free and zero-ledger checks.

Unexpected failures stop the dedicated fixture process without retries; process
exit tears down its test context. That harness is not a production owner or a
model loader. No full-model generation, throughput/load testing, service restart,
power setting change, or other modality migration belongs to this milestone.
