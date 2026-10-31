# Decisions worker

Hikari routes Decisions exclusively to the Python `lfm3_decision` worker with the
fixed `d1-2b-cpu-f32` checkpoint. The public request has no model selector or
feature flag. The FIFO task serializes this fixed model identity for the worker;
no Laya executor, catalog entry or fallback remains. Kadan's language loop still
uses the separate language capability. See [Decisions](../../user-expectations/hikari/decisions.md).

The existing `hikari-worker-decision` Compose service runs the Python worker with
`HIKARI_EXPERIMENTAL_BACKEND=d1`, two CPU cores, 16 GiB RAM and no swap or GPU
reservation. It deliberately does not inherit the GPU worker anchor. The shared
`model_data` volume and old native build volume identity are retained; changing
workers does not remove downloaded files or historical database records. The
verified d1 store keeps its existing `experimental/<model>/<revision>` path.
Acquisition still requires an explicit license-acknowledged Settings download;
evaluation never acquires artifacts. Native stores retain ownership of the
standalone transcription-formatting preference.

Existing installations must rebuild the API, frontend and Decisions worker
together; an old native worker cannot serve the new fixed route. No database or
model-file migration is performed. Previously verified d1 downloads are reused
at the same path; otherwise Settings requires an acknowledged download before
evaluation. Retired Laya files stay inert on disk.

The pinned checkpoint is `LiquidAI/d1-2B` at revision
`10e7671d44e4b37700020b1eed3f594c5f2af0b4`, under the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-2B/blob/10e7671d44e4b37700020b1eed3f594c5f2af0b4/LICENSE).
The verified catalog includes its executable Python modules, loaded only from
local files by Transformers with `trust_remote_code=True`. The disposable child
uses CPU F32, two Torch threads and no external hub kernels. Importing the adapter
or transport loads no inference library. Transformers progress bars are disabled
before loading because their process-shared locks can otherwise leave named
semaphores behind after executor termination.

The typed adapter maps Choice, Score and Noul, validates returned identities,
types and probabilities, then publishes answers in request order. Score rubrics
require 2–10 levels in both the public schema and the worker adapter. Unsupported
sizes fail explicitly rather than being compressed, reinterpreted as numerical
Estimate tasks or redirected to another model. `score_readout="mean"` preserves
the fractional expected zero-based ordinal index; the upstream default median
would change that contract. Removing the public trial selector and rejecting
formerly supported wider Laya rubrics are intentional API compatibility changes.

Before forward execution, the pinned engine encodes its actual text sequences.
The adapter sums state, questions, options/rubrics and template/special tokens
across every sequence and rejects totals above 65,536 without truncation. This
ceiling is not measured capacity; the architectural context is 128,000 tokens.
The existing 8,000 state-character, eight-question and 16,000 serialized-character
limits remain in force.

The scheduler grants a fixed minimum and preferred 15 GiB host RAM and zero
VRAM before spawning. The worker reports the smaller of physical free RAM and
its container ceiling, with 1 GiB headroom. This grant covers the measured
short-input F32 initialization peak; it neither estimates 64K memory needs nor
grows with input size. Extra host RAM does not enlarge it. Runtime RSS polling
terminates overruns, but is not a hard OS allocation limit; operator container
and host limits remain necessary. The 180-second deadline includes model loading,
not time waiting for admission. Cancellation, disconnect, memory overrun and
deadline failure terminate and join the process before a terminal result or grant
release. Repeated cancellation cannot skip cleanup; late results are fenced out.

Automated validation uses model-free unit fakes only. Earlier authorized isolated
manual runs used the same pinned model and existing dependencies, two CPU cores,
16 GiB RAM, no swap, GPU or network access. Eight questions across all three types
at 334 encoded tokens completed in 154.23 seconds including loading, with sampled
peak executor RSS of 12,322,725,888 bytes. In a later seven-job long-lived Runtime,
repeated early cancellations settled in 0.096–0.142 seconds and an evaluation
cancellation in 0.114 seconds. Exactly 1,024 tokens entered evaluation but timed
out in 180.12 seconds without an answer. Subsequent invalid Score requests failed
before loading, confirming queue recovery. Every job confirmed executor exit,
scratch cleanup and grant release; no named semaphores, resource-tracker warnings
or OOM events remained after the progress-bar fix.

These are bounded CPU lifecycle measurements, not HTTP/production-hub acceptance,
a quality comparison, a performance gain over another model or a GPU prediction.
No 64K inference, memory-fit or completion benchmark was run. This default switch
was validated without inference, acquisition, deployment or service changes.
