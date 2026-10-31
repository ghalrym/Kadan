# Opt-in d1 CPU decision worker

Hikari still serves Laya through its normal decision worker. The d1 trial is a
separate CPU worker, enabled only with `HIKARI_ENABLE_D1_DECISIONS=true` on the
API and worker. Launch the existing Python worker image with
`HIKARI_EXPERIMENTAL_BACKEND=d1`; no Compose service or default assignment changes.
The decision request must explicitly include `model_id: "d1-2b-cpu-f32"`.
Omitting it continues to route to Laya, even while the trial is enabled. The
regular Decisions UI does not select d1. See [Decisions](../../user-expectations/hikari/decisions.md).

The worker advertises its pinned catalog only while enabled. Acquisition uses
the existing explicit, license-acknowledged download action and verified model
store; evaluation never acquires artifacts. Importing the adapter or transport
loads no inference library. The admitted disposable child loads only local files,
disables external hub kernels and uses CPU F32 with two Torch threads. The pinned
engine's exact text sequences are encoded before inference, rejecting more than
1,024 total tokens rather than relying on its batching budget or truncating.

The candidate is the original `LiquidAI/d1-2B` checkpoint at revision
`10e7671d44e4b37700020b1eed3f594c5f2af0b4`, under the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-2B/blob/10e7671d44e4b37700020b1eed3f594c5f2af0b4/LICENSE).
Its weights and tokenizer are data; the checkpoint also supplies executable
Python modules loaded by Transformers with `trust_remote_code=True`. Reviewing
or acquiring the data is separate from executing that code.

The adapter maps Choice, Score and Noul questions and validates the returned
identities, types and probabilities before publishing answers in request order.
It requests `score_readout="mean"`: d1's default median would change Hikari's
fractional expected-score contract. Explicit d1 requests reject Score rubrics
outside two to ten levels before queueing. The pinned official JSON validator
and single-digit Score readout support that range only; its broader numerical
Estimate task is not an equivalent ordinal rubric. Ordinary Laya requests keep
their existing broader validation. No request is
truncated, answered synthetically or redirected to another model.

The d1 executor disables Transformers progress bars before loading. Their tqdm
process-shared lock can leave named semaphores registered with the long-lived
parent after executor termination. Protocol events provide progress instead,
including the accepted encoded-token count immediately before evaluation.

The hub reserves 15 GiB host RAM and no VRAM before spawning the executor. This
envelope covers the measured F32 initialization peak below and leaves the
worker's 1 GiB headroom inside a 16 GiB container cap; it is
not evidence that all admitted inputs fit. The runtime monitors executor RSS and
enforces a 180-second deadline including model loading. Cancellation, disconnect,
RAM overrun and deadline failure terminate and join the process before publishing
a terminal result or releasing the grant. Repeated cancellation cannot skip
cleanup. The pure adapter's entry/publication fences are not the interrupt
mechanism: process ownership supplies it. Existing GPU worker behavior is retained.

Automated validation uses fake model responses only. A separate authorized manual
CPU experiment used the pinned original artifacts and existing dependencies:
two CPU cores, F32 parameters, 16 GiB RAM, no swap, no GPU or network access and
a 180-second deadline per process. Encoded sequences were checked against a
1,024-token ceiling before inference; no compilation or external hub kernels
were enabled.

The first run reached its deadline after completing four valid Choice calls over
a 54-token sequence. Loading took 82.98 seconds; the three warm calls had a
16.10-second median. Remaining cases did not complete, and no full-run peak was
captured. A narrower second run completed Choice, fractional Score and Noul in
one 120-token request: 83.65 seconds loading, 16.08 seconds initializing the
decision engine, then 23.52 seconds evaluating. Peak process RSS was 12,319,993,856
bytes; RSS after the response was 8,636,542,976 bytes. These are bounded CPU
measurements, not a quality comparison or evidence of a speedup over Laya.

Both test containers and their processes exited. Downloaded artifacts and evidence
remain in a separate experiment cache; running services and their saved model
stores were untouched. The measured F32 process does not fit the current decision
worker's 3 GiB envelope. The separate 15 GiB CPU trial has model-free tests of
routing, RAM admission, token bounds, deadline and confirmed cancellation cleanup.

A subsequent explicitly authorized manual acceptance used the real local Runtime
FIFO and spawned executors with the same pinned model, two-core CPU and 16 GiB
container caps. Read-only status reported ample physical host RAM; its local
controller checked the container's 15 GiB admissible limit before granting
resources. It never connected to a production API. Cancellation before `loaded`
completed in 0.162 seconds after the
cancel command, and cancellation one second after `loaded` completed in 0.111
seconds. Eight questions across all three types, totaling 334 encoded tokens,
completed in 154.23 seconds including loading. Sampled executor RSS peaked at
12,322,725,888 bytes. An oversized input was rejected explicitly before forward
execution, without truncation. Every terminal event followed confirmed executor
exit and grant release; executor scratch was removed. The container exited and
was removed, with no GPU devices exposed, installs or service changes.
That initial run reported two semaphore objects for resource-tracker cleanup at
shutdown. A model-free diagnostic reproduced a named semaphore surviving a
terminated Transformers progress bar; disabling its progress bars created none.

After the fix, another authorized run reused one Runtime for seven real executor
jobs. Three consecutive cancellations before `loaded` settled in 0.096 to 0.142
seconds; cancellation one second after the evaluation event settled in 0.114
seconds. An exactly 1,024-token request entered evaluation but did not finish
within the unchanged 180-second deadline. Its failure settled in 180.12 seconds
with confirmed cleanup, rather than publishing an answer. Subsequent malformed
one-level and eleven-level Score requests failed explicitly before loading the
model, demonstrating that the same queue recovered. Every job left zero named
semaphores, no executor or scratch files, and a released grant. Shutdown produced
no resource-tracker warning; the container reported no OOM events and was removed.
Sampled peak executor RSS across the run was 12,321,288,192 bytes.

This establishes these bounded local lifecycle cases, not HTTP/WebSocket or
production-hub acceptance, indefinite cancellation endurance, quality or a
speedup over Laya. The full-limit CPU timeout, RAM reservation and narrower Score
contract are explicit resource/product tradeoffs before replacing the default.
The supported official recipe uses F32 on CPU and BF16 on CUDA/MPS; GPU resource
use and performance are unmeasured. Laya remains the default.
