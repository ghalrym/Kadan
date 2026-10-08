# Synthetic mixed stack and whole-token publication

This milestone implements exactly four resident text decoder layers: three linear
attention layers followed by one full attention layer. It adds admitted BF16
embedding lookup, final offset RMSNorm, original NVFP4 vocabulary projection and
deterministic greedy selection. The fixture is intentionally tiny: hidden 16,
vocabulary 16, capacity 8, routed experts 4/top-2, routed width 16, shared width 32.
There is no tokenizer, checkpoint IO, API/UI change, model generation or benchmark.
Synthetic selected IDs are not text, and no synthetic tokens/sec is measured or
presented as actual-model speed.

## Shared private producer and one cursor

`decoder_producer.hpp` extracts the existing decoder's weight uploads, borrowed
buffer views and execution into an internal noncopyable producer. It owns no
memory, admission, pin or cursor. It is permanently bound to its owner's cursor;
foreign/stale steps fail before any CUDA operation. Kernels are unchanged. The
standalone `cuda::Decoder` remains an owning adapter with the same public contract,
now using this producer. Existing CPU golden/runtime tests cover that refactor.

`cuda::Stack` embeds four producers in its charged descriptor object. All four
borrow subregions of one arena and the same `StateCursor(4, capacity)`. No layer
has a private cursor to advance or an independently allocated/reserved child.

A step receives an integer token ID. After embedding lookup, each complete
attention/post-norm/MoE/residual producer runs and acknowledges its layer. The
model then computes final norm, vocabulary scores and greedy selection. It
rounds vocabulary projection results to BF16 before comparing them. Every score
must be finite; exact ties choose the lower ID. The selected ID returns to host,
is range checked, and all status checks, synchronization, cancellation checks
and unpin finish before the prevalidated same-thread cursor commit. Only then
is the trivial `Selection{token,eos}` returned. No CUDA/resource operation follows
publication, and there is no per-step allocation.

`step(input, stop_on_eos, cancelled)` computes a prediction for every input ID.
During prompt feeding, `stop_on_eos=false` allows a caller to ignore predicted EOS;
the result's `eos` still describes that prediction. When `stop_on_eos=true` and EOS
is selected, the owner becomes terminal until reset. The EOS prediction is not
itself consumed into state. A subsequent step after EOS or exhausted capacity is
rejected before execution and does not change progress. The caller feeds chosen
IDs back explicitly; there is no hidden generation loop, sampling or tokenization.

Cancellation is cooperative, checked before work, within/between layers, after
selection and after the final host copy/synchronization. The atomic flag must
remain alive during the call. A cancellation racing after the final check can
complete normally. Bad input IDs and attempted-step failures invalidate the whole
sequence; a late failure never publishes partial layer progress or a Selection.
No rollback is promised. All four state regions and scratch are physically zeroed
and synchronized before reset revalidates the sequence. Diagnostics reject invalid
or stale results. Runtime failures poison the owner until close.

All activations and selection buffers are internal: unlike the standalone decoder,
this API borrows no caller device IO that could be consumed on failure. Uncertain
recovery throws `DeviceBufferQuarantine` and retains the pinned model arena. Close
must prove quiescence before freeing it. Failed close retains conservative charges
and latches against retry; no reset of the GPU or service is attempted.

## Admission and accounting

One `Resources` handle charges the complete RAM descriptor and exact device arena
before allocating the descriptor. Four optional embedded producers do not allocate
on the heap. One `cudaMalloc` holds all layer weights/state/scratch, embedding and
final-norm weights, packed vocabulary head, and final buffers. Successful close
destroys charged metadata immediately even if the wrapper remains alive.

For the verified fixture/toolchain:

- Device arena: **54,784 bytes**, one allocation and one ledger entry. No caller
  device IO is needed, and no per-layer ledger records are created.
- Charged host descriptor: **133,104 bytes**. Source fixture data, caller diagnostic
  buffers, shared resource-manager infrastructure and CUDA context overhead are
  caller-owned/additional. Upload staging is bounded at 2KiB on the stack.
- CPU reference numeric budget: **5,816 bytes**. As with the existing CPU oracles,
  this budget describes numeric state/scratch, not control objects or borrowed
  immutable source weights.

The resident model is one eviction unit. Sharing immutable weights between sessions,
streaming experts, separate session-state admission and multiple devices are later
boundaries, not implemented by nesting independently admitted owners here.

## Independent CPU validation

`stack_golden.py` is new **test-only Python requiring separate user review**.
It uses only stdlib scalar/matrix equations, no native/production/model imports.
For every requested prefix it recomputes all four layers from immutable token IDs,
using explicit prefix KV matrices and expanded-matrix linear recurrence. It does
not obtain expected values from the incremental coordinator. Existing approved
Python generators are unchanged.

The fixture varies FP8 and expert multipliers across layers. Prompt IDs `[2, 7]`
and three fed-back generated IDs produce consumed IDs `[2, 7, 11, 5, 3]` and
predictions `[5, 11, 5, 3, 3]`. Each step checks all layer outputs, all physical
states, final normalized activations, all vocabulary logits and selected IDs, then
repeats after reset. BF16 values are exact; independent FP32 recurrence uses absolute
error at most `2e-5`. These IDs belong only to the synthetic vocabulary.

CPU tests additionally cover:

- Greedy ties, predicted EOS ignored during prompt feeding, terminal EOS and reset.
- Capacity, bad IDs, pre-cancellation, no result assignment on failed steps.
- A fourth-layer MoE overflow and final vocabulary-head overflow; all state resets.
- Real CUDA ownership code with CPU runtime/kernel stubs: fourth-layer failure
  after every fake state has mutated; cancellation after earlier layers and final
  copy; selected-ID copy/final synchronization errors, bad selection, runtime
  poison, uncertain recovery quarantine, partial construction and latched close.
- Actual descriptor allocation vs ledger while retaining eight closed wrappers;
  insufficient budgets before allocation; 1,023 other residents plus this one
  model; no allocation during successful steps/reset.
- Actual private producer rejection of foreign/stale capabilities before enqueue.
- Kernel-launch counts for ten-step replay (1,150), late fourth-layer failure (99)
  and final-head failure (114).

The runtime shim validates ownership/control flow, not numerical GPU behavior.
All 21 CPU tests pass with ASan/LSan and nonrecovering UBSan. CTest remains
CPU-only. CUDA compilation targets explicit SM86 with parallelism
two and does not probe or execute a GPU.

## GPU plan — unexecuted, review required

The opt-in executable is `kadan-stack-parity --allow-gpu-validation --device 0
--stage N`. It is not registered in CTest. Use separate fail-stop processes after
independent exact-head math/runtime review and stage clearance. Verify head and
binary hash, idle queues, GPU memory/power/temperature baseline, and use one GPU
with a 30-second timeout. Stop on unexpected outcomes without rerun or automatic
stage advancement.

| Stage | Work | Compute launches |
| --- | --- | ---: |
| 1 | One prompt token, all layer/output/state goldens, reset and cleanup | 115 |
| 2 | Two prompt IDs plus three generated-ID steps, reset and replay | 1,150 |
| 3 | Expected fourth-layer activation overflow/invalidation/rejected reuse/reset/pre-cancel; then a separate zero-head greedy-tie/EOS owner | 99 + 115 = 214 |

One successful token uses embedding 1 + three linear decoders 84 + full decoder
27 + final norm 1 + vocabulary projection 1 + selection 1 = 115 launches.
Stage 3's owners run sequentially, closing the first before constructing the
second. Peak admitted device and owner RAM remain 54,784 and 133,104 bytes. Harness
caps are 128KiB device and 512KiB host, plus caller/context overhead. Stages 1/2
use one allocation/free pair; stage 3 uses two sequential pairs. Every owner must
finish with zero ledger charges, invalid closed health and physically zero reset
state. No GPU execution has occurred for this milestone.

## Next work

Pinned tokenizer assets and reference vectors, actual checkpoint role binding and
bounded loading, real-model comparison, multi-device quiescence/publication and
performance validation remain separate. The target model's 50+ decode tokens/sec
goal is unmeasured and is not inferred from these correctness tests.
