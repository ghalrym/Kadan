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

Validation uses fake model responses only. Model compatibility, cold/warm latency,
peak resident memory and cancellation cleanup remain unmeasured. A bounded manual
experiment must precede any default change. The proposed CPU experiment uses the
existing LFM image, two CPU cores, 16 GiB RAM and a 180-second process deadline;
it has no GPU access and changes no running service or saved model store.
