# CPU synthetic reference evidence

Base: merged #112, `28371deb630bbd83cf1015bbbc24d982bd29653b`.
Pinned image/source/backend identities and commands are in [README.md](README.md)
and [backend-pins.json](backend-pins.json). Full local artifacts are retained in
`/tmp/kadan-reference-evidence/run-review`; the small comparison report is committed as
`results.json`. These are generated fixture payloads, not actual checkpoint data.

| Fixture | Exact 0/0 result | Mismatching logits | Expected / HF selected ID | Maximum absolute error |
| --- | --- | ---: | --- | ---: |
| Tiny, H16/E4/top2, four layers | PASS | 0 | 11 / 11 | 0 |
| Representative, H256/E12/top8, four layers | PASS | 0 | 9 / 9 | 0 |
| Tiny with zero router weights | **FAIL** | 16 | 2 / 11 | 0.1015625 |

Tiny capture SHA256:
`8dcfabfcd2af3d3bd64787502d4265789b99de209549894f163f7ef93cbd3717`.
Representative capture SHA256:
`65c1de1736064171754d66ef57930615c22efa73f5de999eb38f7f4b6a80b8e2`.
Tied HF capture SHA256:
`72a8c48c470047ab94e76910dbb500d88dd087f35b48ee9b535411634794050f`.

This runner compares HF against independent equations; it does not execute a new
native forward. Canonical native conformance is separate from exact pinned HF
compatibility (see README). The tie case retains the independent lower-ID router
contract. The recorded HF `torch.topk` run
chose `[2,3]` at every layer for equal probabilities; the equations choose
`[0,1]`. Expected final-token margin is 0, HF margin is 0.015625; maximum FP32 ULP
distance is 15,204,352 and maximum BF16 step distance is 232. These numbers are
diagnostics only. **The full fixture runner exits 1.** No changed tolerance or
expected-failure convention turns that parity failure into a success.

Each successful capture process exits 0 independently of the subsequent parity
comparison, reports one forward, three chunk/conv calls, zero recurrent/update
calls, cache length 1, BF16 logits and zero final isolated logical reservations.
Effective threads are 2 intra-op / 1 inter-op. No GPU or actual-model run occurred.

Validation of the test infrastructure: 14 stdlib tests pass; all 23 native CPU
CTests pass under ASan/UBSan with bounded build parallelism 2. Initial sanitizer
execution inside the ptrace sandbox failed because LeakSanitizer cannot operate
there; the approved run outside that sandbox passed without disabling sanitizer
checks. Earlier wrapper bring-up errors (optional tokenizer metadata, dictionary
cache states) were fixed before these numerical results. The first attempt to
retain evidence hit a scratch-directory UID mismatch before fixture creation;
the final run uses the host UID and a private scratch directory.

This establishes two one-token synthetic exact matches and one unresolved tie
failure, not actual-model parity, native GPU performance, longer-prefix cache
correctness or a safe maintenance procedure. The runtime from #112 is unchanged.

The existing native comparator independently returned 0 for tiny/representative
at 0/0 and 1 (`capture_token_or_eos_mismatch`) for the tie. The sanitizer rebuild
replaced the local comparator, SHA256 now
`fac64d9bbb418760e75360a42dc71103421cbd309c0f2c1b1c56791877f5620f`.
The old comparator hash is historical, not a current execution pin. The CUDA
correctness executable remains unchanged at
`0c2e7fef0f7b2911b83d7ab6c68415049436d33268d6d27b391907be0d02331c`.
Any held actual-model execution must review/pin the final binaries again.

Review follow-up reran all three fixtures with exact cache layer/slot cardinality,
shape and dtype validation. Captures and comparison results are byte-identical
to the prior run. The 14-test stdlib suite now covers missing/extra cache layers,
empty/extra/wrong state slots, absent/wrong-shaped/wrong-dtype tensors, and
TimeoutExpired raw stdout/stderr preservation (including invalid UTF-8), with
failure retained in results.json. All capture Python source hashes in the new
reports match this revision; no expected capture or tolerance changed.
