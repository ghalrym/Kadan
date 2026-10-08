# Shared generation scheduling: foundation and follow-up MRs

Baseline: merged MR114, master `4e6f2cd8a2fe8b9d2f4d4891b1995a519dfe5fd5`.
The original deployment checkout and its untracked files are preserved; this
work uses an isolated checkout. No service restart, model read, GPU execution,
database write, or full image rebuild is needed for this foundation.

## Existing architecture and image status

- `api/memory_manager/queue.py`: bounded Redis arrival-order queue, one consumer,
  process lock, cancellation and cleanup ownership across modalities.
- `api/inference/resources.py`: Python RAM/per-device admission, leases, eviction
  and model handoff. It is still the production scheduling authority.
- `native/include/kadan/resources.hpp` and `native/src/worker.cpp`: C++ ledger and
  accounting-only protocol, including an image workload tag, not image execution.
- `native/src/model_worker.cpp` and `api/inference/llm/native.py`: persistent Qwen
  token worker. Offload reaps the worker and releases RAM and VRAM; it is NOT RAM
  parking. The CUDA model arena combines weights, request state/KV and scratch;
  checkpoint rows stream into VRAM. Reported live throughput of approximately
  51.92 decode tokens/sec is prior evidence, not remeasured by this MR.
- Master's image adapter explicitly raises UnsupportedFeature. Image HTTP routes
  and queue entries alone do not imply a working provider.
- There IS unmerged image execution work: open PR52, `ai-reed/native-qwen-image`,
  inspected local branch commit `739263677f854d2372115f37323221acc390abbd`, contains
  a Python Torch/Diffusers `QwenImage21Pipeline` adapter with CPU parking and
  shared lifecycle wrappers. Its use of the word “native” does not mean original
  Kadan C++ image kernels. Other open image proposals include PR53–57 and PR67.
  This is code evidence, not validation of those PRs or real image inference.
  An independently reviewed/rebased image adapter can participate via the
  existing Python pipeline before implementing C++ diffusion execution.

## MR sequence

1. **This MR:** bounded, single-owner C++ generation FIFO/lifecycle component,
   reusing Resources, with CPU tests. Distinct executors acknowledge load,
   execution and physical cleanup. No production transport or executor wiring.
2. Separate immutable weight backing from disposable request state. Implement
   GPU residency, RAM retention under an explicit byte budget, and cold-tier
   reload. Count source + destination + pinned staging + workspace transition
   peaks before copies. Prefer unchanged checkpoint files as cold backing;
   only derived formats need new cache files. Bound derived SSD/NVMe cache bytes,
   use atomic publication, versioned keys, checksums, eviction and failed-write
   cleanup. Disk is slow storage requiring explicit reads, not extra fast RAM;
   mmap alone is neither a RAM budget nor a latency guarantee.
3. Add worker transport and native Qwen executor lifecycle integration, preserving
   token protocol compatibility. Test process death, cancellation during copies,
   synchronization and cleanup acknowledgement before reservation release.
   `cuda::Model` is construction-thread/device affine: route all lifecycle work
   through its owning executor thread. The synchronous reset/step/close protocol
   cannot receive cancellation during a long step; add a separately serviced
   cancellation signal or supervised leaf termination, then acknowledge cleanup.
4. Separate API/Python MR: bridge Redis arrival order to the native coordinator;
   define exactly one admission owner and prevent duplicated independent ledgers
   from overcommitting. Integrate/review the existing image provider separately,
   retaining its distinct pipeline. C++ image execution is an additional MR.
5. Separate UI MR for queue, residency and cache state.

## Foundation contract

`GenerationQueue` is called from one serialized event loop. Arrival order means
successful `submit` order, not order of racing client clocks. It admits only
text/image requests, returns bounded monotonically increasing IDs, and never
moves a same-model request ahead of a different model. A model key must include
checkpoint identity and execution configuration. Workload and footprint must
also match for reuse. The footprint is a conservative full allocation peak;
per-request reservations and host retention are intentionally future work.

Poll returns an outstanding action until acknowledged: transport adapters must
launch it once. Load success transitions to execute; completed reusable requests
retain the resident model. A different head request requires cleanup first.
Failed/partially cancelled loads and failed execution require cleanup too.
Adapters must synchronize request work before `completed`, and acknowledge
`cleaned(..., true)` only after every owned allocation and asynchronous operation
is gone. Failed cleanup remains charged and blocks the queue; retries must be
idempotent. Cancellation signals an active executor but never pretends to free
its memory. Stop rejects submissions, drops pending work and drains active work
and residency through the same acknowledgements. Stale acknowledgements fail.

This class owns accounting, not physical allocations; destruction cannot perform
executor cleanup. Its owner must drive stop/drain before destroying the executor.
The eventual process supervisor must enforce timeouts and reconcile physical
process exit. This MR does not promise RAM parking, disk cache, real image
execution, durability, multi-producer thread safety or live service integration.

## Validation and development

Build only affected CPU targets and their dependencies; use existing source/hot
reload workflow for subsequent API development. Do not routinely rebuild images.

```
cmake -S native -B /tmp/kadan-generation-build -DCMAKE_BUILD_TYPE=Debug -DKADAN_ENABLE_CUDA=OFF
cmake --build /tmp/kadan-generation-build --target generation-queue-tests resource-tests model-worker-tests kadan-worker --parallel 2
ctest --test-dir /tmp/kadan-generation-build -R '^(generation-queue|resources|model-worker-protocol|worker-protocol)$' --output-on-failure
```

Tests cover mixed FIFO, same-model reuse, configuration changes, queue bounds,
queued/active/load cancellation, partial-load failure, failed cleanup retry,
stale acknowledgements, shutdown and reservation conservation. Existing native
CI automatically builds and runs this test with ASan/UBSan; CI evidence belongs
to the exact pushed commit, not merely to a matching local working tree.
