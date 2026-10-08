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
worker. `python` remains the default; `native` remains protocol 1. `KADAN_GPU`
still selects auto or a single index. Native execution remains one configured
small Qwen checkpoint. The child owns Qwen residency in an exclusively dedicated
CUDA process; never place unrelated allocations there. The child ledger cannot
detect CUDA allocations made outside its ownership, and close resets its context.

Parent ResourceManager reservations subdivide as follows:

| Owner | Admitted envelope | Release condition |
| --- | --- | --- |
| Native host | 260 MiB loader/control + 256 MiB retained tensors + 512 MiB Python/tokenization | Child reaped and tokenizer released |
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
