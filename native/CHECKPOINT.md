# Bounded shard-local checkpoint reader

This original Linux C++20 reader builds on the quantized projection reference in
#99. It opens one explicitly selected safetensors shard, validates all its tensor
metadata, and reads only requested projection rows and their companion scales.
It does not route through the model index, load the full model, initialize CUDA,
or change the worker/API execution path.

## Actual checkpoint inspection (2026-10-07)

Read-only inspection of the existing installed
`small-1355db6a052410cfd62085d94b58866fd0f2c3c5` checkpoint found:

| Shard | Header bytes | Tensor count | File bytes |
| --- | ---: | ---: | ---: |
| model-00001-of-00003.safetensors | 6,937,600 | 51,662 | 10,006,877,608 |
| model-00002-of-00003.safetensors | 8,566,488 | 63,484 | 10,003,595,752 |
| model-00003-of-00003.safetensors | 1,253,352 | 9,322 | 3,413,864,960 |

The index is 13,726,227 bytes with 124,468 entries. Config identifies ModelOpt
0.37.0 and MIXED_PRECISION. Inspection read only file sizes, bounded JSON, and
headers, with a 16 MiB index cap and 12 MiB per-header cap after reading sizes.
No actual checkpoint tensor payload was read and no model execution occurred.

The observed dtypes are `F32`, `BF16`, `F8_E4M3`, and `U8`. Example shapes:

- Layer 0 `linear_attn.in_proj_qkv`: FP8 weight `[8192,2048]`, F32 scale `[]`.
- Layer 0 expert 0 and shared expert `gate_proj`: U8 weight `[512,1024]`,
  FP8 block scales `[512,128]`, F32 `weight_scale_2` `[]`.
- `lm_head`: U8 weight `[248320,1024]`, FP8 scales `[248320,128]`, scalar F32 global.

Each sampled projection's companion tensors are co-located in its shard. This
motivates a useful first reader without an additional 124k-entry index parser.
Missing or cross-shard companions fail with `missing_tensor`; future index
routing must validate index-to-shard associations and share one memory budget.
The metadata inspection informed scope; the native reader itself has been tested
on tiny synthetic files, not run over the production checkpoint.

## Usage and ownership

Build with the standard CPU-only [CMake commands](README.md). Metadata-only CLI:

```sh
/tmp/kadan-native/kadan-checkpoint-inspect /path/to/checkpoint model-00001-of-00003.safetensors 67108864
```

The last argument is an explicit allocator budget in bytes. Defaults cap a header
at 16 MiB and a shard at 65,536 tensors, above the measured per-shard maxima. The
CLI prints tensor count and retained accounted metadata bytes. It never reads a
tensor payload. No new dependencies or package installation are needed.

Native callers create a shared `MemoryBudget`, open `Shard(root, basename,
budget)`, and call `load_modelopt_rows(prefix, first_row, row_count,
payload_budget)`. The returned move-only `Projection` owns the selected packed
weight bytes, block scales, and decoded aligned float multipliers. `view()` binds
those buffers to #99's `quantization::Matrix`; the projection must outlive every
view and operation. Buffers survive reader destruction and remain charged to the
shared budget until their final owner is destroyed. Views cannot be obtained
from temporary projections. Do not use moved-from projection views.

`MemoryBudget` is a thread-safe PMR allocator quota, not a second machine-wide
resource manager. Its limit must come from an already admitted RAM envelope.
Header buffers, parser strings, tensor tables, and projection vectors use it,
including temporary growth allocations. Failed reads, validation failures, and
allocation failures release their charged buffers. Fixed object/control-block,
file-descriptor, stack, allocator bookkeeping and OS page-cache overhead are not
counted; admission must retain headroom for those. Payload limits bound logical
returned tensor bytes independently of the shared allocator ceiling. These are
allocation-request accounting limits, not RSS or OS memory guarantees.

## Validation and restricted schema

The reader follows the [safetensors format](https://github.com/safetensors/safetensors/blob/main/README.md#format):
little-endian header length, relative tensor offsets, row-major storage, complete
payload coverage, and exact byte counts. It checks overflow before shape/offset
arithmetic, rejects duplicate names/fields and unknown tensor fields, and detects
holes, overlaps, out-of-file data and truncated reads. It accepts scalars and
zero-sized tensors in metadata. Projection row requests must be nonempty.

The schema parser is original, iterative and deliberately narrow. Supported
headers use ASCII strings (including ASCII JSON escapes), unsigned decimal
integers, the four observed dtypes, rank at most 8, and strings at most 512 bytes.
`__metadata__` accepts at most 1,024 unique string-to-string entries. Unicode
beyond ASCII, unknown dtypes, arbitrary nested metadata, malformed escapes and
numeric alternatives such as negative dimensions fail closed. There is no
unbounded recursive parser and no general-purpose JSON dependency. This is not
a promise of compatibility with every valid safetensors file.

Binding explicitly asserts the ModelOpt convention; the caller must establish
that convention from trusted checkpoint configuration. The reader does not infer
producer identity from similarly named tensors, parse `config.json`, or support
compressed-tensors inverse scales, swizzled data or activation quantization.
NVFP4 requires U8 `[rows,K/2]`, FP8 `[rows,K/16]`, K divisible by 16 and a scalar
F32 global scale (`[]` or `[1]`). FP8 requires a 2D weight and F32 scalar or row
scales (`[]`, `[1]`, `[rows]`, `[rows,1]`). Only selected scale/weight values are
validated through #99's reference contract, without allocating decoded weights.
Unused tensor payloads are neither read nor numerically validated. BF16 is
recognized for metadata coverage, not executable by this projection loader.

## Filesystem and mutation boundaries

The caller supplies a trusted root directory. Untrusted shard names must be a
single bounded ASCII basename, with no slash, NUL, `.` or `..` traversal. The
root directory and shard leaf are opened with `O_NOFOLLOW`; the shard uses
`openat`, must be a regular file, and is retained through an RAII descriptor.
`O_NONBLOCK` prevents a FIFO from hanging before the type check. No whole-file
mapping is used, so truncation yields an error rather than a mapping fault.
Ancestor components of the caller-supplied root are trusted, not sandboxed.

Size, modification time, and change time are rechecked after metadata parsing
and around payload reads. Path replacement cannot redirect the retained file
descriptor. These checks detect ordinary changes but are not an immutable
snapshot or cryptographic integrity guarantee: checkpoint files must remain
immutable during use. Concurrent malicious writers and checkpoint provenance
verification are outside this milestone.

## CPU verification and remaining work

Tiny temporary-file tests cover actual dtype/shape conventions, row slicing,
reference decode/matvec, payload and allocator exhaustion/recovery, ownership
past reader lifetime, JSON/schema failures, duplicate fields/tensors, size
arithmetic overflow, rank/string/header limits, holes/overlaps/coverage, nonfinite
selected values, wrong scale layouts, traversal, symlinks, FIFO/directory files,
and post-open truncation. Tests never touch Postgres or production model files.

Next steps are cross-shard index routing, explicit global admission integration,
checkpoint identity/integrity policy, BF16 tensor loading and original CUDA
kernels. No GPU correctness, whole-checkpoint native compatibility, generation,
throughput, deployment, or other-modality migration is claimed here.
