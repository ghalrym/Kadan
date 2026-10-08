# API bridge for ordered text and image execution

Based on merged native MR118 (`5cc3d58fd7b042dcbddb8ce395e004d943912d87`).
Reuses PR52's image backend at `739263677f854d2372115f37323221acc390abbd`:
QwenImage21Pipeline, pinned checkpoint catalog, downloads, image generation/edit
routes and atomic PNG/result manifests. UI work is excluded; only the schema and
required generated client change.

## Ownership and opt-in migration

Python remains the global intake/admission coordinator: the Redis queue's accepted
order is authoritative and one consumer executes it. After validating the queued
payload, model availability, device configuration, deterministic capacity and
admitted edit decoding, MemoryManager awaits other feature wrappers'
physical parking before dispatch. A later cache hit cannot pass an older image.
No second Python residency table or competing native global queue is introduced.

Set `KADAN_LLM_BACKEND=native-resident` to use protocol 2 with the installed MR118
worker rebuilt for cache planning schema 3 and the cache telemetry command. `python` remains the default; `native` remains protocol 1. `KADAN_GPU`
still selects auto or a single index. Native execution remains one configured
small Qwen checkpoint. The child owns Qwen residency in an exclusively dedicated
CUDA process; never place unrelated allocations there. The child ledger cannot
detect CUDA allocations made outside its ownership, and close resets its context.

Parent ResourceManager reservations subdivide as follows:

| Owner | Admitted envelope | Release condition |
| --- | --- | --- |
| Native host | 514 MiB loader/cache-control + planned packed text bytes + 512 MiB Python/tokenization | Child reaped and tokenizer released |
| Native model GPU | Planner's full arena | Verified `parked` acknowledgement, or child reaped |
| Native context | 512 MiB on the selected GPU | Dedicated child reaped; never released by park |
| Image host | Twice checkpoint bytes plus 8 GiB construction/workspace | Pipeline references cleared and cleanup confirmed |
| Image model GPU | Checkpoint bytes plus 8 GiB, or 8 GiB sequential offload workspace | Sync, CPU transfer/hook removal and allocator cleanup confirmed |
| Shared Python image context | 512 MiB per selected GPU | Process lifetime; not evictable by tensor parking or pressure |
| Image input/output staging | 1 GiB while decoding/generating/publishing | Request exits |

These are conservative admission envelopes, not measured allocation totals.
The native child receives only its admitted host/cache and GPU/context sub-budget;
Python-owned memory is excluded from the child limit. Consecutive text requests
keep the full parent model envelope conservatively while native frees/recreates
request state. On handoff, native keeps its process and bounded immutable RAM
backing. Restore reserves the model arena before sending another start command.
Pressure may close/reap the idle child and discard backing to recover capacity.
The image executor remains Torch/Diffusers, with its own execution pipeline.

The 64 GiB native cold limit bounds registered checkpoint references, not a physical
SSD directory. No duplicate model disk cache, swap-as-fast-RAM, or managed SSD
storage tier is added. Full native image execution and a fully native global
scheduler remain later work.

## Identity, cancellation and uncertain cleanup

Every protocol-v2 response must match the child session nonce and request ID;
request IDs must increase. Restarted children establish a new session. Each
admitted request submits, starts, steps and ends synchronously through that leaf.
Queued cancellation stays in the global queue. Active text cancellation terminates
and reaps the child with the existing bounded supervisor policy; a cancellation
command never waits behind a blocked step. A missing park acknowledgement
quarantines reuse and retains the parent envelopes until close/reap reconciles it.
Failed reap retains all three reservations.

Image cancellation is observed at the pipeline's step callback. The queue waits
for the worker thread to exit and cleanup before starting the next request. A
failure removes unpublished output; successful output is atomically renamed.
Uncertain image synchronization/cleanup retains accounting and rejects reuse.
Torch's shared context remains charged even after model close because
`empty_cache()` does not destroy it.

## Deterministic validation

`api/tests/memory_manager/test_resident_bridge.py` uses real isolated Redis, the
real RuntimeManager, real ResourceManager and actual subprocess IPC. Numerical
text/image leaves are synthetic. Barriers assert this exact sequence, with text
D arriving while image B is active:

```text
text:A:start → text:A:end → text:park
→ image:B:start → image:B:end → image:park
→ text:C:start → text:C:end → text:D:start → text:D:end
```

The same native process and host backing reservation survive this handoff; there
is no overlapping model execution. Tests cover queued cancellation without a
spurious park, active native cancellation/reap before image execution, active image
callback cancellation before text, image failure without published output, and
unknown park cleanup blocking image execution and text reuse. Protocol tests
cover stale sessions, wrong/reused IDs, new sessions after restart, malformed
startup, timeouts, unconfirmed startup reap, exact sub-budgets, preserved host
backing and retryable restore pressure. Image tests retain the process context,
prevent spending it twice, and inject failed synchronization.

Local CPU validation after review corrections: full API suite passed (412 tests,
two skips), including the active-native and startup-failure cases.
All six frontend tests passed after schema/client generation. `uv pip check` and
`api.inference.check_install` passed with Torch 2.14.1+cpu, torchvision 0.29.1+cpu,
and PR52's pinned Diffusers revision. This builds no local API image and downloads
no production model weights. CI validates the changed dependency image separately.

Real checkpoint text → image → text acceptance and image quality/performance
remain unrun, requiring the image checkpoint and an explicit deployment budget.
MR118's 192-logit GPU evidence is tiny text-only correctness with **pre-step**
cancellation recovery; it does not prove interruption of an in-flight CUDA kernel.
The live server and its GPU worker are unchanged by this development branch.


Review corrections: history exposes only canonical UUID directories whose
manifest identity matches the directory. A barrier test proves staging results
are hidden before rename. Edit decoding runs before handoff under a 1 GiB
reservation with eviction disabled, so invalid input or insufficient validation
space cannot move resident text. The same immutable input is decoded again under
the execution staging reservation. Invalid device settings and impossible logical
budgets fail before provider construction/execution or parking. The effective edit
limit is the queue's 16 MiB complete JSON payload (slightly under 16 MiB base64,
about 12 MiB decoded), even though the decoder's independent hard cap is 20 MiB.
See [read-only hardware readiness](LIVE-SWITCH-READINESS.md) for existing weights,
cached runtime and exact proposed isolated test budgets.


## Full packed text cache and restore deadlines (MR119 follow-up)

The initial bounded hardware run at `0e84a7b` generated its image, but text C
expired during native `start`: the harness used a 30-second token-step deadline
for a full arena restore. `start` now uses `load_timeout`; actual token steps keep
`step_timeout`. The next approved hardware harness will use 300 seconds for
initial load/restore and 30 seconds per token, retaining the existing overall,
image, thermal and memory guards. No second successful run is claimed here.

`--plan-resident` now returns `plan 3 HOST ARENA VOCAB CONTEXT STAGING PACKED TENSORS`.
The two added fields sum validated text bindings only (packed weights, scales,
and calibration), excluding vision/MTP tensors and GPU state/workspace. The
resident command session remains version 2 with a new idle-only command:

```
cache SESSION 0
cache SESSION 0 CAPACITY RETAINED REGISTERED HITS MISSES HIT_BYTES SOURCE_BYTES EVICTIONS ENTRIES
```

The adapter records an acknowledged snapshot at readiness, end, and park, exposed
through `cache_stats()`. RETAINED is actual application-owned payload bytes;
CAPACITY is reserved maximum payload. SOURCE_BYTES counts successful checkpoint
payload reads, including cache fills; it does not measure physical SSD traffic.
HITS/MISSES count read-through requests based on whether the tensor was already
retained, and HIT_BYTES counts requested bytes served by those preexisting
buffers. Counters saturate at uint64 maximum. Complete cache reuse requires
unchanged SOURCE_BYTES across restore, increasing HITS/HIT_BYTES, and expected
RETAINED—not merely unchanged PID/RSS or zero `/proc/.../io` physical reads.

Default admission prefers the complete planned packed text footprint. A supplied
`cache_ram_bytes` constructor cap allows smaller/zero backing; if the full model
exceeds the total parent host capacity less fixed overhead, automatic admission
falls back to at most 256 MiB. Normal physical/active-reservation admission still
applies; no unadmitted allocation or silent host overrun is allowed. Existing
checkpoint files are the cold storage tier, bounded to 64 GiB of registered
source payload. This does not add a disk cache or treat disk as fast RAM.

The previous 1,024-entry/per-tensor-reservation design could not retain this
124,116-tensor model. Native model backing now uses one aggregate reservation
and a bounded 131,072-entry table, with 256 MiB conservative cache control and
allocator headroom separately admitted. The normal generic per-entry cache
remains available. Actual payload allocations grow on demand under the fixed
aggregate envelope; close frees buffers before releasing that envelope.

Row/chunk loading remains bounded to existing staging. Automatic cache misses
now fill only when capacity is free, without evicting another tensor; otherwise
only the requested slice streams from checkpoint. This prevents alternating
weight/scale/scalar reads from repeatedly evicting/refilling entire tensors.
Explicit generic retain/evict still supports its LRU policy. Full-cache restore
therefore needs no row-order rewrite, and partial-cache fallback avoids read
amplification rather than pretending the entire model stays in RAM.

For existing `small-1355db6a052410cfd62085d94b58866fd0f2c3c5`, read-only shard header
inventory gives 20,825,156,824 packed text bytes in 124,116 tensors. Total checkpoint
payload is 23,407,580,856 bytes; excluded vision/MTP payload is 2,582,424,032 bytes.
No payload was read or model executed to make this inventory.

| Host reservation / transition component | Bytes | GiB |
| --- | ---: | ---: |
| Complete packed text cache | 20,825,156,824 | 19.394939 |
| Native loader/control + Python/tokenizer | 1,075,838,976 | 1.001953 |
| Text host total | 21,900,995,800 | 20.396892 |
| Image host envelope (weights × 2 + 8 GiB workspace) | 74,821,161,408 | 69.682637 |
| Image publication / edit scratch | 1,073,741,824 | 1 |
| Combined transition host peak | 97,795,899,032 | 91.079528 |
| Margin inside 96 GiB parent / cgroup cap | 5,283,316,072 | 4.920472 |

The loader's staging and metadata are included in fixed native overhead. Cache
miss fills allocate the final retained buffer directly; there is no second
full-model temporary copy. Image parameters and transition workspace remain
conservatively included in its existing envelope. Fresh observed host availability
at the first run's end was 424,394,457,088 bytes (~395.25 GiB); the next test must
probe again. GPU budgets are unchanged: 22 GiB parent; final text peak
23,355,836,416 bytes (21.7518 GiB) including both 512 MiB contexts, and sequential
image phase 9 GiB including both contexts. The GPU margin is narrow and the
existing physical probes/guards remain required.
