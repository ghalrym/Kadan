# d1 decision adapter preparation

Hikari still serves Laya through its normal decision worker. The d1 adapter is
an explicitly invoked preparation component, without a catalog entry, worker
registration, deployment setting or change to the public API. It accepts an
already loaded model; importing or testing the adapter loads no model or inference
library. Current behavior remains documented in [Decisions](../../user-expectations/hikari/decisions.md).

The candidate is the original `LiquidAI/d1-2B` checkpoint at revision
`10e7671d44e4b37700020b1eed3f594c5f2af0b4`, under the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-2B/blob/10e7671d44e4b37700020b1eed3f594c5f2af0b4/LICENSE).
Its weights and tokenizer are data; the checkpoint also supplies executable
Python modules loaded by Transformers with `trust_remote_code=True`. Reviewing
or acquiring the data is separate from executing that code.

The adapter maps Choice, Score and Noul questions and validates the returned
identities, types and probabilities before publishing answers in request order.
It requests `score_readout="mean"`: d1's default median would change Hikari's
fractional expected-score contract. The trial rejects a one-level Score rubric,
which the current HTTP schema accepts but d1 does not support. No request is
truncated, answered synthetically or redirected to another model.

Cancellation checks fence entry and publication. They cannot interrupt a blocking
model call or establish memory cleanup. Production integration must retain
Hikari's admitted executor ownership, termination and confirmed cleanup; this
adapter supplies none of those lifecycle guarantees by itself.

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
worker's 3 GiB envelope. Production admission, worker routing and interruptible
executor ownership remain unwired; eight-question, long-input and real worker
cancellation acceptance remain unverified. Laya remains the default.
