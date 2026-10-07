# Native text manifest and explicit placement

`ModelManifest` binds the installed Qwen3.6 mixed checkpoint to native loading and
placement metadata. It reads `config.json`, `model.safetensors.index.json` and
bounded safetensors headers, and validates the complete index against pinned shard
descriptors. It does not read weight payloads, allocate model buffers, probe CUDA,
load a model or change the running Python service during construction/planning.

This is an original C++20 implementation using the existing native shard reader
and resource-budget allocator. No third-party engine, kernels or JSON parser code
is copied, wrapped or vendored. It is a usable prerequisite for a model executor,
not that executor or a full-model memory-fit/performance claim.

## Validated profile and tensor binding

The supported profile is the installed `qwen3_5_moe` /
`Qwen3_5MoeForConditionalGeneration` ModelOpt MIXED_PRECISION text decoder. Native
architecture fields retain layer types, dimensions, expert counts, head counts,
convolution size, context/token limits, RMS epsilon and rotary parameters.
Attention bias, tied embeddings, non-gated Q output, unsupported activation/dtype,
non-default rotary behavior, misaligned dimensions and incompatible layer schedules
fail closed. Dimension/layer/expert limits prevent config-driven unbounded loops.

Expected tensor names and shapes are derived from those dimensions. Every text
role must appear once, with the exact supported dtype/shape:

- BF16 embeddings, final/per-layer norms, routers, shared-expert gates,
  GatedDeltaNet A/dt/norm/convolution and auxiliary a/b projections.
- FP8 gated-Q/K/V/output projections for full-attention layers; FP8 QKV/z/output
  projections for linear-attention layers, with scalar F32 weight scales.
- NVFP4 routed/shared gate/up/down and output-head projections, with packed U8
  weights, per-16-column E4M3 scales and scalar F32 global scales.
- Scalar F32 input calibration scales for every quantized projection. They are
  bound/accounted and can be loaded explicitly, but current native primitives
  remain weight-only; activation calibration is not silently implemented.

`quantized_layers` declarations must match the bound projection encoding/group
size, with no unused declarations. `config_groups` and other optional metadata
are parsed structurally but are not alternative execution instructions; this
milestone does not implement calibrated activation execution. A later model loop
must validate any additional behavioral configuration it uses.

The installed output head is NVFP4, not FP8. Row-scaled FP8 kernels exist, but this
manifest profile deliberately requires the checkpoint's scalar FP8 scale layout;
a different profile needs a reviewed extension. Companions (including calibration)
must share a shard, as they do in the installed checkpoint. Cross-shard companion
assembly is rejected rather than guessed. Quantization **values** are not checked
by metadata binding; the bounded loader and CUDA admission validate them later.
BF16 raw reads similarly do not establish finite numerical values.

Every index entry must exist in its named shard. Unique keys, exact total tensor
count and summed `metadata.total_size` reject missing, extra or misrouted entries.
Each shard independently checks all tensor sizes, offsets and payload coverage.
All text tensors are consumed by roles; unexpected text/top-level tensors fail.
Only `model.visual.*` and `mtp.*` are explicitly excluded from text execution.
Their dtype/offset/byte integrity is checked, but their architecture is not bound
or migrated. Nothing outside the text decoder is claimed executable.

## Quotas, files and loading boundaries

The original metadata JSON parser is ASCII-only, rejects duplicate keys, malformed
numbers/escapes, nonfinite numbers and trailing content, and caps strings at 512
bytes, nesting at 32 and nodes at 262,144 per document. Config/index caps are
1 MiB/16 MiB; defaults allow up to 64 shards and 200,000 indexed tensors, with
16 MiB/65,536 tensors per shard. Callers may tighten these limits. All variable
metadata buffers, DOMs, tensor tables, items and placement vectors use a shared
`MemoryBudget`. Fixed objects, descriptors, stack and allocator/runtime overhead
remain outside requested-byte accounting and require process headroom.

Metadata/shards are opened relative to the trusted root using basenames,
O_NOFOLLOW/O_NONBLOCK, regular-file checks and close-on-exec. Reads use bounded
pread chunks. Metadata is checked before/after reads; shard descriptors stay open
and are checked before planning and payload reads. This is a pinned local snapshot
with mutation detection, not cryptographic provenance verification. Renaming a
path cannot redirect an already-open shard descriptor. No memory mapping occurs.

Items have stable manifest-local indices. `primary_tensor` exposes the original
primary tensor dtype/rank/shape (including dense/conv dimensions and packed weight
shapes) without duplicating the entire shard metadata table. `load_projection_rows` routes a selected
item to its validated shard and the existing row loader. The caller provides a
payload byte cap and may supply a **separately admitted payload MemoryBudget**
for staging or a retained expert bank. Omission preserves the original shared
metadata/payload allocator behavior. Owning projection buffers retain their budget
and survive manifest destruction. `read_dense` writes a bounded byte slice into
caller-owned, caller-admitted RAM; `read_input_scale` reads/validates one scalar.
These operations read payloads only when explicitly invoked, not in the inspector.

## Placement contract

The library requires explicit per-layer device ordinals, an IO device, device
capacities, positive headroom allowances, expert slot counts and host/staging
budgets. No device discovery, eviction, implicit capacity defaults or reservation
occurs. Returned item-to-device assignments refer to the retained manifest.

- Dense/non-routed items are resident on their assigned layer device; embedding,
  final norm and output head use the IO device. Dense byte sizes are padded to
  256 bytes. Quantized resident estimates match the current projection owner's
  aligned weights/scales/input/output/status slab; NVFP4 global scale is by value.
- All routed-expert packed payloads (including calibration scalars) are retained
  in a host bank. Each device reserves the largest gate/up/down expert group
  assigned to it times its explicit slot count (1 through experts-per-layer when
  routed layers are assigned). Slots are a proposed bounded shared cache across
  layers, not actual residency or an eviction implementation.
- Caller-supplied device headroom covers context/driver, layer/state/KV/activation
  and future executor workspace outside those slabs. The planner checks arithmetic
  and capacity but cannot certify that a chosen allowance is sufficient. Context-
  length-dependent state accounting remains a follow-on prerequisite.
- Host total is the full metadata allocator envelope plus all expert backing plus
  one explicit staging pool. Staging must fit the largest complete item transfer,
  including dense embeddings; future finer-grained loaders can reduce this after
  an explicit policy change. Transfers are serialized through this single pool;
  simultaneous independent loads would need additional admitted staging.

Every arithmetic sum/product/alignment is checked; undersized budgets fail before
any model allocation. Planning is not global admission. A future Python supervisor
must grant/retain one host/VRAM envelope from Kadan's existing global authority,
then pass the admitted capacities to the native worker. It must not count these
native plans as a second independently available pool or release an envelope
before cleanup/worker termination. Separating payload and metadata budgets makes
that later integration possible without production Python changes in this PR.

## Actual metadata evidence and CPU verification

The CPU inspector read the actual installed config/index/headers using a 256 MiB
metadata quota. No weight payload was read by it. It found:

| Observation | Value |
| --- | ---: |
| Text layers / routed experts per layer | 40 / 256 |
| Linear / full-attention layers | 30 / 10 |
| Hidden / vocabulary | 2,048 / 248,320 |
| Indexed tensors / excluded vision+MTP tensors | 124,468 / 352 |
| Bound text execution items | 31,333 |
| Full checkpoint payload bytes, including exclusions | 23,407,580,856 |
| Retained requested metadata bytes | 39,605,831 |

With an explicit split after layer 19 (20/20), IO device 0, eight expert slots per
device, 20 GiB device caps, **illustrative 2 GiB headroom per device**, 256 GiB host
cap and 1 GiB serialized staging, the metadata-only plan reported:

| Requested bytes | Device 0 | Device 1 |
| --- | ---: | ---: |
| Non-routed resident slabs | 2,007,803,904 | 703,614,720 |
| Eight expert slots | 14,407,680 | 14,407,680 |
| Caller headroom allowance | 2,147,483,648 | 2,147,483,648 |
| Total planned envelope | 4,169,695,232 | 2,865,506,048 |

Host expert backing is 18,119,639,040 bytes; host total including 256 MiB metadata
and 1 GiB staging is 19,461,816,320 bytes. Largest whole-item transfer is
1,017,118,720 bytes. These are arithmetic plans under the stated policy, not model
allocations, observed execution peaks, sufficient KV sizing or proof of runtime fit.

All eight CPU CTest cases pass with ASan, LSan and nonrecovering UBSan. New tiny
fixtures exercise both attention layer families, all binding roles, multi-shard
index coverage, missing/extra/wrong-format/split companions, quantization conflicts,
malformed/duplicate/deep/oversized JSON, symlink rejection, quota failures, placement
capacity/overflow failures, separate payload quotas, bounded loading and changed
shard detection. No Postgres data, actual-model payload tests or GPU tests are used.

```sh
cmake -S native -B /tmp/kadan-manifest-cpu -DKADAN_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all"
cmake --build /tmp/kadan-manifest-cpu --parallel 2
ctest --test-dir /tmp/kadan-manifest-cpu --output-on-failure
# Metadata-only inspection; ROOT is the trusted installed checkpoint directory.
/tmp/kadan-manifest-cpu/kadan-model-inspect ROOT 268435456
# Explicit two-device planning: host, staging, per-device cap, headroom, slots, split.
/tmp/kadan-manifest-cpu/kadan-model-inspect ROOT 268435456 \
  274877906944 1073741824 21474836480 2147483648 8 20
```

## Remaining gaps

Device-resident layer math, validated recurrent/KV sizing, expert cache execution
and transfers, full native model/decode/sampling, tokenizer integration and the
versioned worker supervisor remain unimplemented. The next useful milestone can
bind CPU layer-state/activation shapes and implement a small original layer
primitive with deterministic references, rather than switching the API prematurely.
Production runtime/LLM adapter/memory-manager changes require a separate reviewed
PR. Existing video/audio/Decisions execution stays unchanged. No full-model
allocation, generation, deployment, service/power change or performance claim is
part of this milestone.
