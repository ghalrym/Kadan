# Next separate API MR: queued text → image → text

Reviewed master after MR117 (`d8d7670327fdd4770250707182f528eeb39c3d3e`) and
open draft [PR52](https://github.com/ghalrym/Kadan/pull/52) at
`739263677f854d2372115f37323221acc390abbd`. This is a proposal, not implemented
API behavior in MR118.

## Existing pieces and missing boundary

PR52 already has real Qwen image execution in `api/inference/image/model.py`:
Torch/Diffusers `QwenImage21Pipeline`, CPU loading, GPU placement or sequential
offload, step cancellation, synchronization and parking back to CPU. It is not a
C++ diffusion executor. `api/services/images.py` validates local pinned model
files and inputs, runs generation, and atomically publishes PNGs plus result.json;
failed unpublished output is removed. Its routes submit through the existing
memory manager. Reuse/rebase these backend pieces; do not recreate diffusion.

The current Redis inference queue can remain the single arrival-order admission
point. Current master dispatches feature wrappers without the explicit outgoing
feature offload loop found in PR52. `ResourceManager` already supports GPU offload
callbacks and `offload_on_handoff=False` reservations, but native protocol 1 closes
its worker and loses backing when offloaded. Process-alive currently means resident
in the adapter; protocol 2 needs separate process/model-residency state.

## Smallest coherent backend change

1. Rebase the image provider, image service/feature, route models, required catalog
   entry and dependency pins from PR52. Verify its pinned Diffusers revision against
   current dependencies. Exclude its UI/browser work. Regenerate OpenAPI and the
   frontend client for changed routes, per repository instructions.
2. Adopt protocol 2 behind the native backend option. Validate session nonce and
   request ID on every response. Keep legacy behavior available. Use the C++ queue
   only as a leaf for already admitted text work; do not build a competing global
   queue that can reorder text and images.
3. Make Python `ResourceManager` the parent admission authority. Divide each native
   worker's admitted envelope into bounded host/cache, GPU model/request arena,
   and persistent CUDA context/headroom. The C++ ledger subdivides that envelope;
   the two ledgers must not each assume independent access to all device memory.
4. Consecutive same-model text requests retain weights and recreate only request
   state. Conservatively retaining the parent arena envelope between them is fine.
   Before the next validated request changes model or feature, await outgoing
   parking/cleanup, then admit its incoming execution. Never skip a queued image
   because later text would hit a cache.
5. On `parked`, release only the parent model arena reservation. Keep bounded host
   backing and context/headroom (`offload_on_handoff=False`). Reserve arena again
   before restarting text. Release context and host only after verified `closed`
   teardown or confirmed child reap and cleanup. RAM pressure may fully close a
   parked leaf. A live Torch process also has a context: `empty_cache()` is not
   context destruction; account its persistent envelope separately.
6. Retain the existing image pipeline in RAM when budget allows; offload its GPU
   tensors before returning to text. Image outputs remain filesystem artifacts,
   with no fixtures or demo data inserted into Postgres. Persisted cache-reference
   limits are not physical SSD quotas; a managed bounded SSD tier remains another
   native MR, with eviction and owned-file cleanup.

Active native cancellation terminates/reaps the supervised leaf with a deadline;
queued cancellation removes work without bypassing other requests. Image cancellation
uses the pipeline callback and confirmed cleanup. Unknown cleanup blocks reuse or
handoff until reconciled, even if the logical request is already cancelled.

## Required evidence in that MR

Use a deterministic queue event trace for text A → image B → text C: B starts only
after A parks, C only after B parks, and no execution overlaps. Verify consecutive
text avoids reload, C reuses retained RAM, strict FIFO under new arrivals, queued
cancellation, active cancellation, malformed/stale/restarted worker identities,
image failure and atomic-output cleanup. Assert context envelopes cannot be spent
by the incoming model, host retention respects budget, and failed cleanup cannot
release an uncertain reservation. Keep tests isolated from Postgres.

A separate opt-in real text/image/text acceptance run needs the actual image
checkpoint and a reviewed GPU/RAM budget. MR118's synthetic text parity run does
not establish image correctness or production performance. A fully native global
scheduler with an image executor registry remains later work; this transitional
API MR provides real ordered cross-modality execution without conflating the
Torch image pipeline with native diffusion.
