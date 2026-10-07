# Native quantized projection reference

This milestone adds original C++20 decoding and CPU matrix-vector execution for
the canonical ModelOpt NVFP4 and FP8 weight formats needed by Kadan's pinned
`nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint. It is a numerical reference for future
loaders and original SM86 kernels, not a production inference backend.

## Evidence and scope

The pinned [NVIDIA configuration](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/blob/1355db6a052410cfd62085d94b58866fd0f2c3c5/config.json)
identifies ModelOpt 0.37.0, MIXED_PRECISION, FP8 projections, and NVFP4 groups of
16. Kadan's existing `read_projection` contract uses byte-packed NVFP4 weights,
E4M3 block scales, and a scalar `weight_scale_2` multiplier, or FP8 weights with
scalar/per-row `weight_scale`. Those are the explicitly supported native views.
The config also contains activation quantization settings; this reference does
**not** emulate them or establish full-model accuracy/parity.

Format arithmetic is derived from NVIDIA's [NVFP4 description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
and [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), with synthetic
hand-derived golden vectors. No Strata/FreeToken implementation or kernel is
copied, wrapped, vendored or ported. Existing Python code remains unchanged.

## Contracts

`kadan/quantization.hpp` exposes `e2m1`, `e4m3fn`, and explicit little-endian FP32
scalar decoding. E2M1 has magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6. E4M3FN uses bias
7, supports subnormals and signed zero, reaches 448, and reserves 0x7f/0xff for
NaN (not infinity). The low nibble is the first column of each packed FP4 byte.

`Matrix` is a non-owning, contiguous row-major view. Its spans must remain alive
and immutable during calls. The future loader must validate safetensors dtype,
shape, byte offsets, bounds, and tensor associations before constructing a view;
this library does not parse a checkpoint manifest or open model shards. Do not
reinterpret unaligned file bytes as floats: decode scalar data with `fp32_le`
into owned aligned storage, and keep that storage alive with the view.

| Encoding | Weight bytes | Block scales | Multipliers |
| --- | --- | --- | --- |
| ModelOpt NVFP4 | rows × columns / 2 | rows × columns / 16 E4M3FN bytes | exactly one positive finite FP32 scalar |
| ModelOpt FP8 | rows × columns E4M3FN bytes | none | one positive finite scalar or one per row |

NVFP4 columns must be divisible by 16. Every span must have the exact required
size; logical shape multiplication is checked for overflow. Empty matrices,
unknown encodings, NaN weights, and nonfinite/negative block scales fail closed.
Zero block scales are valid. Multipliers must be strictly positive and finite.
Compressed-tensors inverse scales/int32 packing, swizzled layouts, expert-bank
3D views, and FP8 block-scaled layouts are deliberately unsupported. A future
expert selector may construct a validated 2D view for one expert.

`decode_rows` expands only a selected row range. `matvec` reads packed weights
directly without creating a dense matrix, and returns one value per row. Both
require an explicit output byte budget, allocate only their returned vector,
and return no partial result on errors. The input spans' ownership and memory
reservation belong to the caller; the output must be reserved separately before
calling. Validation scans scales (and FP8 weights) on every call; this is a
correctness reference, not an optimized execution loop.

Weight values are computed from the exact small-format value and scales using
double intermediates, then rounded to FP32. Matvec uses those FP32 weights and
FP32 inputs with double accumulation, then rounds the output to FP32. It rejects
nonfinite input, decoded-weight overflow and output overflow. Underflow follows
ordinary IEEE FP32 conversion. It implements neither BF16 rounding nor activation
quantization, bias, batching, tensor cores, or CUDA reduction order. Future GPU
kernels need an explicit numerical tolerance and hardware validation; this
reference does not imply bitwise parity with a GPU backend.

## CPU validation

Use the standard native CMake/CTest commands in [README.md](README.md). The new
`quantization` test runs tiny CPU fixtures only and checks all 16 E2M1 codes, all
256 E4M3FN codes, all 256 packed-byte combinations, signed zeros/subnormals,
little-endian scalar decoding, multi-row/multi-block scaling, independently
calculated dot products, malformed spans, overflow, NaN handling, and exact
output budgets. The native CI workflow automatically runs this suite.

Next: a bounded safetensors reader with shard/index validation and dtype-to-view
binding, then original packed projection kernels. Full checkpoint loading,
worker integration, GPU numerical correctness and 50+ tok/sec remain unverified.
No GPU workloads, benchmark, deployment, or running-service changes are part of
this milestone.
