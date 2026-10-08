# Immutable RAM backing and checkpoint cold tier

This follows the design of [MR116](https://github.com/ghalrym/Kadan/pull/116),
and is rebased onto merged MR116 at master
`b73f37d27d67ac76a5407acce096190f35a3306c`. It adds a native library, not a
production executor or API change. No live GPU work or container rebuild is
required. It is original Kadan code, not Strata/FreeToken code.

## Actual storage and limits

`WeightBacking` owns real immutable tensor byte arrays, populated by the existing
bounded safetensors `Shard` reader. RAM entries survive destination GPU cleanup;
LRU eviction physically frees host arrays before releasing their reservations.
An entry larger than the configured RAM cache stays cold. Retention under shared
RAM pressure evicts only this store's cached weights, never another owner's
request state. If space still cannot be obtained, `retain` returns false.

Cold backing is the original open checkpoint descriptor and tensor identity.
`cold_limit` bounds the sum of registered tensor payload bytes, including entries
also in RAM; the entry limit bounds metadata cardinality. Duplicate registrations
under distinct keys are conservatively charged again. No new disk files are
created, so additional disk allocation is zero. This quota does NOT bound the
size of user-owned checkpoint files, filesystem page cache, or the model-download
catalog. Forgetting an entry never unlinks a checkpoint. Cold reads use bounded
`pread` chunks; disk remains a slow I/O tier, not mapped extra RAM.

Keys must include model revision and tensor identity. The store retains shared
ownership of a validated Shard descriptor and checks its unchanged-file contract
before/after retention/copy. A changed/truncated checkpoint fails closed, even if
RAM bytes exist. This reuses Shard's stat-based change detection; it does not add
cryptographic authenticity checks. Checkpoints and the trusted root must remain
immutable during their registered lifetime. Replacing a pathname does not retarget
an already-open descriptor. Reconstructing a store registers the original file
again; there are no partial cache files or redundant copies to recover.

No transformed format is needed for this raw tensor backing. If a later executor
needs transformed cold data, add a separately bounded derived cache with versioned
keys, content validation, atomic publication, disk-full recovery and eviction.
Do not silently add unbounded temporary files or overwrite source checkpoints.

## Accounting and ownership

Use the same `Resources` instance as the coordinator/executor. Each retained array
has a real RAM reservation. Shard/parser metadata uses its caller-supplied
`MemoryBudget`; key/map/control overhead must fit the caller's admitted control
and metadata envelope, outside these payload counters. The existing global
Resources resident-count limit also applies; this is not a process RSS limiter.

The transfer lifecycle is:

1. `begin_transfer(key, destination)` atomically reserves destination bytes plus
   cold-read staging, while retained source RAM remains charged. Only one target
   (host or one GPU) is allowed; GPU capacities cannot be summed. The caller must
   not allocate the destination until admission succeeds. Scratch and KV/state
   use separate reservations in this same ledger; they cannot enter the weight
   cache. Failed admission has no destination ticket or hidden destination buffer.
2. `copy(ticket, sink)` sends bounded const spans from retained RAM or from real
   checkpoint reads. The sink must consume/synchronize before returning; spans
   cannot escape or be retained by asynchronous device work. Lifecycle operations
   run on the owning executor thread (including CUDA construction-thread/device
   affinity). Reentrant store mutation is rejected. Cancellation is checked between
   chunks and at completion, not during a blocking read or sink call.
3. On success, `commit(ticket)` frees staging, shrinks the reservation to the
   destination footprint, marks it resident, and transfers ledger ownership to
   the executor. The executor then owns pin/unpin and acknowledged eviction/free.
   Retained source arrays remain independent and reusable.
4. On cancellation, failed copy, or failed destination allocation, destroy and
   synchronize the partial destination before `cleaned(ticket, true)`. A false
   acknowledgement keeps the full transition peak charged and blocks new store
   operations. Failed copy cannot be retried or committed on the same ticket;
   cleanup then a new transfer is the recovery path. No partially copied tensor
   may be published. Calling cleanup is the owner's physical proof obligation.

Example payload accounting for an 8-byte tensor and 3-byte read chunk:

| Source / destination | During transfer | After commit |
| --- | --- | --- |
| Cold / GPU | RAM 3, GPU 8 | RAM 0, GPU 8 |
| Retained / GPU | RAM 8, GPU 8 | RAM 8, GPU 8 |
| Retained / RAM | RAM 16 | RAM 16 |

The store supports one transfer at a time and serial calls on one executor
thread. It does not create threads, issue CUDA calls, or physically allocate the
external destination. Destroying it with an unacknowledged transfer deliberately
leaves the destination reservation charged in the shared ledger, even though its
owned staging is freed. Supervisors must retain the ledger and reconcile such
quarantined reservations only after physical cleanup. Explicit commit/cleanup is
the normal path; destruction is not a recovery protocol.

## Integration sequence and images

Next native integration must connect this backing to the validated model loader
and split the current CUDA arena's immutable weights from KV/state and scratch.
Raw tensor bytes still require the existing quantization/finite-value validation
and binding transformations before execution. This MR does not claim the live
Qwen worker already parks in RAM or accelerates reloads; performance requires that
integration and actual-model measurement. It also does not yet wire MR116's queue
to this shared ledger. Preserve exactly one scheduling/accounting authority when
bridging Python in its separate MR.

Master's image feature remains unsupported. Existing open
[PR52](https://github.com/ghalrym/Kadan/pull/52) contains a Torch/Diffusers
`QwenImage21Pipeline` adapter (inspected branch commit
`739263677f854d2372115f37323221acc390abbd`) with CPU parking and lifecycle wrappers.
Review/rebase/integrate that existing pipeline separately rather than recreating
it. It is not original C++ image execution; native diffusion would be another MR.
Both modalities can eventually use the same scheduling and residency authority
while preserving distinct execution pipelines.

## Validation

Only affected native targets and dependencies are built locally:

```
cmake -S native -B /tmp/kadan-weight-build -DCMAKE_BUILD_TYPE=Debug -DKADAN_ENABLE_CUDA=OFF
cmake --build /tmp/kadan-weight-build --target weight-backing-tests checkpoint-tests resource-tests --parallel 2
ctest --test-dir /tmp/kadan-weight-build -R '^(weight-backing|checkpoint|resources)$' --output-on-failure
```

The three suites also run with `-fsanitize=address,undefined
-fno-sanitize-recover=all`. LeakSanitizer must run outside the tracing sandbox.
Native CI runs the full CPU suite with those sanitizers on the exact commit.

Tests use only temporary local safetensors files, never Postgres/model data.
They verify real byte equality for retained and cold sources, LRU order, bounds,
external RAM pressure, transfer peak rejection, separate request ownership,
const chunk limits, queued transfer exclusion, stale tickets, reentrancy,
cancellation before/during transfer, sink failures, source mutation mid-read,
allocation failure after admission, failed-cleanup retry, successful ownership
handoff, abandoned-ticket quarantine and cold-store reconstruction. No CUDA
performance or correctness claim is made by these CPU tests.
