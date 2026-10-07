# Native inference worker foundation

This is original Kadan C++20 code. It does not copy, wrap, vendor or port
Strata/FreeToken code. The first milestone implements an accounting library and
a standalone development worker boundary, not model inference. There are no
CUDA kernels or production API adapters. The separate
[bounded checkpoint reader](CHECKPOINT.md) now loads selected rows from one
explicitly chosen safetensors shard into owned quantized projection buffers.
A separate [quantized projection reference](QUANTIZATION.md) now provides original
CPU NVFP4/FP8 format decoding and matrix-vector execution for future loaders and
CUDA kernels. The worker executable does not call that library yet.
Existing Python/API/UI execution is unchanged; other modalities are not migrated.

## Build and test without GPU access

```sh
cmake -S native -B /tmp/kadan-native -DCMAKE_BUILD_TYPE=Debug -DKADAN_ENABLE_CUDA=OFF
cmake --build /tmp/kadan-native --parallel 2
ctest --test-dir /tmp/kadan-native --output-on-failure
```

Requirements: CMake >= 3.24, a C++20 compiler, threads, and Python 3 for protocol
tests. Tests allocate only tiny CPU accounting structures. No CUDA detection,
CUDA driver initialization, inference, benchmark, or database access occurs.
`KADAN_ENABLE_CUDA=ON` optionally checks the CUDA toolchain and defines the
`kadan_cuda` interface target for future kernels. It is not linked by the current
worker and has not been validated here. Future RTX 3090 kernel builds must set
`CMAKE_CUDA_ARCHITECTURES=86`; NVFP4 checkpoint encoding does not imply hardware
FP4 arithmetic support on SM86. Kernel design and GPU validation are later work.
Do not run GPU workloads until Andrew explicitly authorizes them. Do not restart
or deploy services as part of these development commands.

## Boundary version 1 (development control plane)

Launch `kadan-worker HOST_BYTES [GPU_BYTES ...]` with explicit unsigned decimal
budgets, one per device (at most 64 GPUs). Budget position 0 is RAM; subsequent
positions map to worker-local CUDA ordinals. Budgets are accounting ceilings,
not memory probes, allocations, or a guarantee that external memory is free.
Both the library and executable allow at most 64 device budgets. The ledger
holds at most 1024 residents, including zero-byte, loading, and evicting records.
A full ledger rejects reserve with `resident_limit`; a slot is reusable only
after cleanup acknowledgement. These fixed limits bound control-plane storage
independently of the allocation budgets. No implicit defaults or pooled VRAM. Zero device arguments permits host-only use.

The parent writes one ASCII whitespace-separated command per newline to stdin;
stdout returns exactly one `ok ...` or `error CODE` line. Startup failures use
stderr and exit 2. EOF exits; an unterminated final command is processed. The
parent must serialize commands; no unsolicited events, request IDs, concurrent
streams or retry/deduplication guarantee are implemented. A lost reply requires
terminating this development process and starting a fresh session, not replaying
an unknown mutation. Input frames are capped at 4096 bytes; oversized frames are
drained through the next newline and rejected without changing accounting.

| Command | Reply / purpose |
| --- | --- |
| `hello` | `ok kadan-worker 1 accounting-only` |
| `reserve WORKLOAD HOST_BYTES [GPU_BYTES ...]` | `ok HANDLE`; exact budget shape required |
| `loaded HANDLE` | Loading completed; resident becomes eligible for execution |
| `pin HANDLE` / `unpin HANDLE` | Protect/unprotect active resident; multiple pins supported |
| `evict HANDLE` | Begin eviction only if resident and unpinned |
| `eviction_failed HANDLE` | Restore resident eligibility while keeping its reservation |
| `released HANDLE` | Acknowledge cleanup of a failed load or an evicting resident |
| `snapshot` | `ok RESIDENT_COUNT USED_HOST_BYTES [USED_GPU_BYTES ...]` |

All successful mutations other than reserve return `ok`. Workloads are `llm`,
`image`, `video`, `speech`, `tts`, and `decision`, matching Python resource labels.
These labels are extension points only. Handles are monotonically increasing and
never reused within a process. They are session-local and must not survive worker
restart. Unknown or released handles fail closed. Reserve is atomic across all
budgets; subtraction-before-comparison prevents unsigned overflow. Loading and
evicting residents still consume their full reservation. Partial physical
cleanup must keep the full reservation until all associated allocations are gone.
`eviction_failed` assumes the executor has restored a usable resident; otherwise
it must keep the resident evicting until cleanup succeeds.

This executable owns no allocations. Its cleanup commands simulate executor
acknowledgements for development tests; do not expose the pipe as a public API.
In the eventual production worker, only its trusted executor may acknowledge
load completion or physical cleanup, after synchronizing relevant device work.

## Integration architecture and next milestones

`api/inference/resources.py` currently owns cross-workload memory reservations,
while `api/inference/feature.py` keeps cancellation pending until actual cleanup.
Preserve those guarantees. Do not run this ledger beside Python with each
claiming the entire machine's capacity. During migration Python remains the
global admission authority: it will reserve a bounded host/per-device envelope
for the worker, and this C++ ledger will subdivide only that envelope. Python
must retain the envelope until worker cleanup/termination is confirmed. Existing
image/video/audio/Decisions adapters continue using Python's global manager.
A later explicit migration can move global ownership into the worker; there must
always be one global authority. Allocator/context overhead and loading scratch
must fit inside granted budgets, with conservative headroom.

The resource state machine is loading -> resident -> evicting -> removed.
Loading failure can go directly to removed after cleanup. Failed eviction can
return to resident. Active pins block eviction; the eventual scheduler must own
pins through completion/cancellation and release them only after device work
finishes. The library serializes accounting with a mutex; execution and cleanup
must occur outside it. It deliberately does not call callbacks or choose victims.
The future scheduler will select inactive residents, request eviction, await the
executor acknowledgement, then retry admission. Host offload will reserve a
separate destination before transfer and release the source only after success;
this milestone does not implement transfer or automatic LRU eviction.

The next focused PR should add an opt-in Python supervisor and bounded versioned
request/event framing (request IDs, cancellation, shutdown, crash handling),
using synthetic CPU executors to test envelope ownership and cleanup. Keep API
behavior unchanged until the LLM adapter has parity. Then add an original
checkpoint manifest/loader with validated tensor bounds, RAII host/device buffers,
and original SM86 packed-weight projection kernels with CPU numerical references.
Build layer placement, expert residency, batched dispatch, recurrent attention,
KV management, and graph-safe scheduling on that foundation. Do not import an
external inference implementation to fill gaps.

The target is `nvidia/Qwen3.6-35B-A3B-NVFP4` at 50+ decode tokens/sec on two RTX
3090s. No throughput, GPU numerical correctness, CUDA build success, memory-fit,
or other-modality migration is claimed by this milestone. CPU accounting tests
cannot establish any of those results. GPU tests/benchmarks require separate
explicit authorization after the breaker trip.
