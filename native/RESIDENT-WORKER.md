# Opt-in resident Qwen worker integration

Based on merged MR117, master `d8d7670327fdd4770250707182f528eeb39c3d3e`.
This is a native integration slice. Legacy `--serve`/protocol 1 remains the
production Python adapter's path; no API, Python, UI, deployment or image
provider changes are included.

## Physical transitions implemented

The optional `ModelOptions::split_residency` path allocates immutable GPU weights
and request KV/state/scratch separately. It preserves the validated logical
layout: each layer's immutable prefix ends at its MoE scratch boundary, with the
remaining mutable suffix in a request allocation. Global embedding/norm/head
weights and global work buffers follow the same partition. Existing kernels
receive remapped pointers; no numerical kernels or quantization rules change.

- Initial load reserves host metadata/control/staging and both GPU allocations
  before allocation/upload. Every reservation shares one Resources ledger.
- `end_request()` synchronizes and physically frees request state/scratch,
  releases its reservation, and invalidates progress. GPU weights remain resident.
- `begin_request()` on resident weights allocates only state/scratch, rebinds
  pointers while preserving projection scalars, zeroes state, and starts at token
  zero. It does not upload weights again.
- `park()` also synchronizes/frees GPU weights while preserving a separate
  context/headroom envelope on the long-lived host/model reservation. Metadata, checkpoint descriptors and bounded immutable RAM arrays
  remain available. No KV/state enters the RAM cache.
- `begin_request()` after park reserves the complete GPU destination before
  uploading through the existing validated Qwen loader. Dense/projection reads
  now pass through WeightBacking: retained tensors come from RAM; others use
  checkpoint reads. Existing finite-value and format validation still runs.
- `close()` removes all owned GPU/host allocations, calls `cudaDeviceReset()`
  successfully, then releases the context/headroom envelope and verifies cleanup.
  Split mode owns its dedicated worker context exclusively: it rejects another
  counted GPU owner at construction or teardown rather than reset their memory.
  Failed synchronization/free poisons reuse and keeps uncertain reservations;
  no success acknowledgement is emitted. The supervisor must terminate/reconcile
  a quarantined leaf rather than repeatedly claiming that cleanup succeeded.

The loader cache registers at most 1,024 tensors, within an added 2 MiB admitted
control envelope; metadata/keys do not silently consume the payload budget.
It evicts its own least-recently-used RAM entries for both byte pressure and
Resources' resident-slot pressure, including room for upcoming GPU reservations.
Other owners' state is never evicted. Entries outside the cache's cold/reference
quota read directly through the validated source. This means the quota bounds
registered cache references, NOT the full checkpoint/model catalog or a physical
SSD cache directory. No additional disk files are created. Bounded cold storage
uses existing checkpoints, never an unbounded transformed duplicate or mmap RAM.

The bank is private to one manifest lifetime; shard ordinal/tensor ordinal keys
cannot alias another checkpoint. Replacing/changing a source fails its existing
Shard identity checks. Raw retained tensors remain immutable, and source RAM,
loader staging and GPU destination coexist under the same ledger during reload.

## Opt-in executable and protocol 2

```
kadan-model-worker --plan-resident ROOT CAPACITY METADATA_BYTES
kadan-model-worker --serve-resident ROOT DEVICE CAPACITY HOST_BYTES GPU_BYTES HEADROOM_BYTES CACHE_RAM_BYTES CACHE_COLD_BYTES
```

The metadata-only plan returns `plan 2 HOST ARENA VOCAB CAPACITY STAGING`.
HOST includes the added control envelope but excludes optional retained payloads;
HOST_BYTES must also leave the desired cache RAM available. CACHE_RAM_BYTES is an
upper bound; insufficient free budget keeps tensors cold. GPU_BYTES still needs
ARENA plus HEADROOM_BYTES. Original `--plan` and `--serve` are unchanged.

At startup the worker emits `ready 2 SESSION VOCAB CAPACITY`. SESSION is a fresh
128-bit OS-random hexadecimal incarnation. Every command must echo it and a
request ID. All acknowledgements carry both fields; IDs are never reused in a
session. Restarted workers have a new session, so old commands cannot act on a
new request with the same small integer ID.

| Command | Reply / meaning |
| --- | --- |
| `submit SESSION 0` | `queued SESSION ID`; bounded FIFO, configured Qwen only |
| `start SESSION ID` | `started SESSION ID`; only the FIFO head can start |
| `step SESSION ID TOKEN STOP` | `token SESSION ID TOKEN EOS COMMITTED` |
| `end SESSION ID` | `ended SESSION ID`; request memory physically released |
| `cancel SESSION ID` | `cancelled SESSION ID`; queued request only |
| `park SESSION 0` | `parked SESSION 0`; model allocations freed, context envelope retained |
| `close SESSION 0` | `closed SESSION 0`; all reservations verified zero |

Engine lifecycle and CUDA work execute on the main owning thread, including
construction/device affinity. The coordinator's zero-byte reservation is a
lifecycle token; the executor and cache reserve physical envelopes in that SAME
ledger, preventing duplicate independent admission. This uses MR116's coordinator
but does not yet provide a model registry or multi-model worker switch. It serves
one configured Qwen checkpoint; image submissions are not advertised or faked.
The coordinator's existing mixed-workload FIFO tests remain applicable to later
registered executors.

## Cancellation and restart boundary

Active cancellation MUST use supervised leaf termination, not a command queued
behind `step`. The resident executable restores SIGTERM's default disposition.
The owning supervisor signals its unreaped child (prefer pidfd where available),
enforces a deadline and uses SIGKILL if needed, then reaps it and reconciles
physical cleanup before releasing its external reservations. The native process
cannot emit a `closed` acknowledgement after being killed. EOF/read errors alone
are not proof of physical cleanup. Replaying interrupted work is a caller choice,
never automatic. Protocol/session IDs prevent stale logical acknowledgements;
they do not themselves prove a dead GPU context has been reclaimed.

This MR tests the process boundary with a CPU leaf blocked inside step: SIGTERM
ends it without waiting for command processing, and no token is published.
It does not change the existing Python supervisor or claim a new Python timeout
policy. Adoption of v2 and multi-model/image orchestration remains a separate MR.
Open PR52 already contains the unmerged Torch/Diffusers Qwen image pipeline;
review/rebase it for later integration rather than recreating it as C++ diffusion.

## Verification limits

CPU fake-CUDA tests execute the actual C++ split allocation, upload, request
reuse, park/reload and failure code against synthetic safetensors. They verify
request-only allocation on consecutive requests, no repeated weight upload,
retained host bytes across park, slot pressure, failures/quarantine, and zero
accounting on successful close. Protocol tests cover FIFO, bounded admission,
queued cancellation, stale sessions/IDs, cleanup failure and owning-thread calls.
Legacy model/decoder/protocol tests remain in the sanitizer CI suite.

The changed CUDA-facing C++ translation units compile against real CUDA headers
with warnings as errors. A tiny real-GPU test matches 192 logits bit-for-bit with
legacy across reuse, park/reload and cancellation recovery; see
[exact synthetic GPU evidence](RESIDENT-GPU-RESULT.md). No production benchmark,
new model download, database write or full image rebuild is part of this MR.
Production-scale validation and API adoption remain required before rollout.
The [separate API proposal](TEXT-IMAGE-INTEGRATION.md) audits PR52 and describes
the smallest real queued text/image/text bridge. The reported live 51.92 tokens/sec
is prior legacy-worker evidence, not a measurement of this mode.
