# Hybrid sequence-state storage

This focused milestone adds native batch-one sequence-state layout and lifecycle
on top of the text manifest. It preserves the distinction between Qwen3.6's
linear and full-attention layers; it does not implement attention, convolution,
GatedDeltaNet recurrence, normalization, model execution or generation yet.
No third-party inference engine/kernel implementation is copied or ported.

## Physical layout

The profile follows the installed config's BF16 activation dtype and FP32 SSM
state, with state dimensions checked against the installed architecture/cache
declarations and the existing Python context estimator. Native storage is an
explicit canonical layout rather than a copy of a third-party cache class:

| Layer kind | Persistent state |
| --- | --- |
| Full attention | Separate BF16 K and V arrays `[token_capacity, kv_heads, head_dim]`. Grouped KV head count is used, not expanded query heads. |
| Linear attention | BF16 raw projected-QKV convolution history `[2*key_heads*key_dim + value_heads*value_dim, conv_kernel]`; FP32 recurrent state `[value_heads, key_dim, value_dim]`. |

A future producer must preserve BF16 rounding before storing activation history,
use the configured key/value head grouping, and apply the correct gated delta
and convolution equations. Byte storage tests do not establish that mathematics.
RoPE-transformed full-attention keys and each producer's token append semantics
remain responsibilities of the future layer executor.

The CPU planner accepts an explicit sequence capacity, layer/device map and
per-device budgets. Shapes, kinds, grouping, capacities, sums, products and
256-byte alignment are checked. Every region is disjoint within its device slab;
linear state is fixed size, while full-attention KV grows with the admitted
capacity. Plans are calculations, not admission capabilities. No dynamic cache
concatenation or full-model buffers are allocated by planning.

For the inspected 40-layer config (30 linear, 10 full; 20/20 split), CPU golden
fixtures verify these exact **state-only** per-device totals:

| Sequence capacity | Bytes on each device |
| --- | ---: |
| 65,536 | 703,528,960 |
| 262,144 | 2,716,794,880 |

The latter already exceeds the illustrative 2 GiB headroom used in #103's sample
plan, before other workspaces or context overhead. These are fixed canonical
layout costs, not measured peaks or proof that a full runtime fits. Future global
admission must combine state, weights, expert residency, activation workspace and
context overhead without treating a previously granted headroom pool as new
capacity or double-counting the same reserved bytes.

## Progress, cancellation and residency

`StateCursor` is the CPU sequencing reference. It issues opaque tokens binding a
non-reused process-local owner identity and a monotonic local generation. Cursor
copying and moving are disabled. Owner identity is assigned atomically and
saturates on exhaustion; object destruction/address reuse cannot revive a delayed
token. Reset never resets the generation. The cursor requires exactly one
completion acknowledgement per layer, refuses incomplete commits and capacity
overruns, and rejects foreign and stale tokens.
Single-token progress is the current boundary; batching/prefill chunk operations
remain future work.

`cuda::SequenceState` implements a **single-device** owner with one persistent
slab, reserved through shared `Resources` before CUDA allocation. It recomputes
its plan from architecture and resource budgets rather than trusting a mutable
public plan. Constructor initialization zeroes and synchronizes the slab before
marking it resident. It verifies the creating thread, current device and SM86;
it does not select/reset devices or change power settings. Fixed bounded control
objects and driver/context overhead need caller-provided host/device headroom.

`begin()` pins residency. Active-step views expose bounded device-pointer regions
for the trusted native producer. Full-attention views expose the current append
slot and a history length including that slot: write K/V before reading that new
position. Linear views expose the current convolution/recurrent buffers in place.
The producer must enqueue its writes on the legacy default stream before calling
`written()`; this acknowledgement is not proof that bytes were initialized.
Raw pointers are borrowed only for the active step and must never escape or be
used after commit, abort, reset or close. Commit synchronizes before publishing
logical progress and removing the pin. There is no per-token allocation.

**Abort invalidates the entire sequence until reset.** In-place recurrence cannot
be rolled back by decrementing a token counter. The owner synchronizes and unpins,
but refuses further steps or state reads until a physical zero/reset succeeds.
This deliberately discards prefix reuse after partial cancellation; it does not
pretend to preserve the last committed prefix. A future conversation adapter must
respect this contract or implement explicit snapshots/replay before integration.
Existing Python conversation behavior is unchanged by this PR.

CUDA runtime failures poison the owner; pending pins remain until cleanup can
establish quiescence. `close()` synchronizes, unpins, marks eviction, frees, then
acknowledges release. Uncertain synchronization/free failures retain the reservation
and latch cleanup failure without automatic retries. Callers retain the resource
manager and must treat uncertainty as fatal worker state. Wrong-thread/device
caller misuse is rejected; all normal methods/destruction belong to the creating
thread with its original device current.

Idle `read_bytes` is a bounded diagnostic copy into caller-admitted RAM. It is
not a production host offload API. Multi-device atomic progress, state transfer,
snapshots, prefix reuse after cancellation, producer failure injection and full
layer numerical integration are not implemented. The CPU planner can describe
multiple devices, but the current CUDA owner places all its layers on one device;
a later coordinator must establish cross-device completion before global progress.

## Verification and staged GPU plan

All nine CPU CTest cases pass under ASan, LSan and nonrecovering UBSan. New fixtures
verify independent tiny-layout offsets, split-device counts, actual-profile size
formulas, malformed/oversized shapes, budgets and cursor commit/abort/reset/stale-
handle behavior. Both CUDA projections and the new state owner/smoke executable
compile for explicit SM86 with the installed CUDA 12/GCC 12 toolchain. Compile
parallelism is two. CTest still contains CPU tests only.

`kadan-state-smoke` is compiled but **not executed**. After independent review of
the exact head, the proposed single-device test creates one **1,280-byte** arena
under a 64 KiB requested-storage budget, checks initial zeroing, performs two tiny
hybrid state writes/commits against independent golden bytes, partially writes a
third step, verifies abort invalidation, resets to zero, checks stale IDs and
explicit/idempotent release. Host fixture buffers are below 64 KiB; additional
CUDA context/module overhead is unmeasured. It uses bounded CUDA copies/memsets,
not a new arithmetic kernel, model files, generation, timing loops or a benchmark.

Before any authorized run: verify reviewed head/binary and idle request/queue
status, inspect read-only power/temperature/workload telemetry, and use one 3090.
Stop on unexpected runtime errors/instability with no automatic rerun. Telemetry
and a storage pass cannot establish electrical safety or throughput. No GPU test
is part of this PR's completed validation until that review/run occurs.

```sh
cmake -S native -B /tmp/kadan-state-cpu -DKADAN_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all"
cmake --build /tmp/kadan-state-cpu --parallel 2
ctest --test-dir /tmp/kadan-state-cpu --output-on-failure
cmake -S native -B /tmp/kadan-state-cuda -DKADAN_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/kadan-state-cuda --parallel 2
# Only after review clearance; intentionally excluded from CTest:
# /tmp/kadan-state-cuda/kadan-state-smoke --allow-gpu-validation --device 0
```

## Separate fully resident expert placement follow-up

The #103 planner **cannot express full expert residency**: every routed expert
is charged to host backing, and each device receives a shared slot pool capped
at experts-per-layer, not storage for all experts assigned across its layers.
Even generous device budgets do not turn this policy into resident placement.
That is a limitation of the first policy, not an architectural requirement to
offload this packed checkpoint.

A separate reviewed placement PR should add explicit `fully_resident`,
`host_cached` and optionally fit-checked `auto` policies. Resident mode must charge
all assigned packed expert storage plus required execution workspace, combine
this state plan with other headroom, and permit host backing release only after
upload completion is acknowledged. Failure/eviction paths must preserve an owner
or a reloadable pinned-checkpoint source until cleanup is certain. Auto mode must
report its chosen policy and assignments rather than silently offload. This PR
does not alter #103 or change that policy. Full-model fit and the dual-3090 50+
decode tok/sec goal still require staged runtime evidence later.
