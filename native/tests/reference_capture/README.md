# Synthetic CPU reference capture (draft)

This standalone test wraps the existing Kadan weight-only Qwen adapter and the
pinned installed Hugging Face PyTorch implementation. It makes one direct text
model call with one synthetic token and a fresh cache of effective capacity 1.
It changes no API, UI, native kernel, service configuration or production module.
It is not an actual-model runner: dimension bounds reject the real checkpoint
before reading its shards or importing Torch. There is no override.

The separate draft follows merged #112 (`28371deb630bbd83cf1015bbbc24d982bd29653b`).
Actual checkpoint payload access, real-model reference inference, GPU execution
and maintenance operations still require separate review and authorization.

## Exact results and unresolved tie behavior

The tiny and representative four-layer fixtures match the independent one-token
stdlib equations exactly (absolute/relative tolerance **0/0**). The all-zero
router fixture does **not** match: installed HF selects experts `[2,3]`, whereas
the independent/native contract selects `[0,1]`. All four layers have a boundary
tie; all 16 final logits differ. This is an unresolved semantic difference,
not a passing parity result. No equation, router or tolerance was changed to
hide it. See [RESULTS.md](RESULTS.md) for the recorded evidence.

`run_fixtures` retains captures, fixture hashes, stdout/stderr and JSON diagnostics
and returns **1 if any comparison fails**, including the known tie case. The
small stdlib regression suite checks format/error handling; its passing status
does not mean HF/native numerical parity passed. The CMake/CI test
`reference-artifacts` runs only that suite. No Torch dependency or GPU execution
is added to ordinary native CI.

## Reproduction

From the repository root:

```sh
python3 -B -m unittest native.tests.reference_capture.test_artifacts
```

The numerical fixtures require the exact locally installed image in
`backend-pins.json`; another wheel with the same version is not an approved
substitute. Use a disposable CPU-only container, no model volume, a read-only
repository mount and a writable scratch directory. The following runs only
miniature generated checkpoints; it intentionally exits 1 for the tie case:

```sh
mkdir -m 700 /tmp/kadan-reference-evidence
# Use a fresh evidence directory for each run.
docker run --rm --user "$(id -u):$(id -g)" --network none --cpus 2 --memory 4g --memory-swap 4g \
  --pids-limit 128 --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,size=512m \
  -e CUDA_VISIBLE_DEVICES= -e USE_HUB_KERNELS=NO -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/work:ro" -v /tmp/kadan-reference-evidence:/evidence:rw \
  -w /work --entrypoint python \
  sha256:985fc03e9ad5fee38fc50b5f088305c9ab94c9a7b3ac53addf2b20622f7c4e4e \
  -B -m native.tests.reference_capture.run_fixtures --out /evidence/run
```

The runner's child timeout is 90 seconds; the wrapper has a 60-second fail-stop
alarm (configurable only between 1 and 120 seconds). An outer container supervisor
is still needed for any proposed actual-model run. These are observation and
escalation deadlines, not guaranteed cleanup/restoration times. Fixtures have a
128 MiB isolated host ledger, a 4 MiB request reservation and no device budget;
container memory enforcement is independent of that logical ledger.

Compare one-token captures independently:

```sh
python3 -B -m native.tests.reference_capture.compare EXPECTED ACTUAL
```

The format is #112's little-endian capture v1, restricted here to one record:
header `[0x4b4d4331,1,vocab,1]`, record `[input,selected,eos,1]`, finite FP32
containers for BF16 logits, trailer `0x444f4e45`. Greedy logit ties use lowest ID.
The parser rejects wrong counts, bounds, progress, nonfinite values, inconsistent
greedy selection, missing DONE and trailing bytes. Diagnostics report mismatch
count, maximum absolute error, FP32 ULP/BF16 step distances and top-token margins.
They never override exact comparison or record equality.

Outputs use exclusive creation, mode 0600, short-write handling, fsync and close.
The wrapper completes backend cleanup and the diagnostic file before writing the
capture and DONE. **Process exit 0, completed diagnostic and matching capture hash
are all required**; bytes containing DONE alone cannot establish success after
fsync/close errors or timeout. Existing paths/symlinks are never overwritten.
Zero reservations means the isolated logical ledger is empty, not a measurement
of allocator RSS; process exit releases the CPU process's remaining allocations.

## Pinned source/backend audit

`backend-pins.json` records the image ID, exact installed source hashes, relevant
AST function spans/hashes and unchanged Kadan adapter/module hashes. The wrapper
checks installed package/source hashes before numerical imports and records its
own Python source hashes in every successful diagnostic. Runtime build is
`torch 2.14.1+cu130`, git `5c4886908584029761b579af026dcfb627c84070`, CUDA build
13.0; execution is CPU only. Transformers is 5.17.0, safetensors 0.8.0 and
accelerate 1.10.1. Build configuration and CPU parallel information are captured.
The image itself supplies the binary provenance; source/version checks alone
cannot attest arbitrary replacement shared libraries outside that image.

`USE_HUB_KERNELS=NO` disables hub decorators, **but does not disable installed
FLA/causal_conv1d fallback selection**. In the isolated test process,
`inspect.unwrap` selects the installed original PyTorch functions for
`causal_conv1d_fn`, `causal_conv1d_update`, `torch_chunk_gated_delta_rule` and
`torch_recurrent_gated_delta_rule`. Their module/file provenance is checked and
reported. No implementation or kernel is copied or vendored. Transparent call
counters require exactly one model call, three chunk calls and three convolution
calls for these four-layer fixtures; recurrent/update calls must be zero.

The audited installed model file is
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`, SHA256
`42b6ca1cd9a2f0754c3dec97f0708dfa7e97e1b6edd9fac88f3139e4e36316c2`.
Relevant rounding and call paths in that pinned file:

| Function/path | Audited behavior |
| --- | --- |
| RMSNormGated (line 220) | FP32 normalization; cast to BF16 before weight multiplication; FP32 SiLU gate multiplication; final BF16. |
| causal_conv1d_fn (272) | Grouped causal convolution in weight dtype, SiLU, result cast to input dtype. |
| chunk gated delta rule (303) | Q/K/V, beta and decay computations in FP32; L2 normalization, chunk length 64, triangular/matrix prefix equations; FP32 recurrent state; output cast to BF16. |
| recurrent gated delta rule (435) | Later cached decode path; deliberately not exercised by this fresh one-token reference. |
| GatedDeltaNet.forward (552) | BF16 projections and beta sigmoid; FP32 decay from A_log/dt_bias; fresh-cache chunk path. |
| eager_attention_forward (727) | BF16 attention matmul/scaling, FP32 softmax cast to BF16, BF16 value matmul; gated output projection. |
| TopKRouter (884) | BF16 router linear; FP32 softmax/topk and selected normalization; selected weights cast BF16. HF topk tie ordering is preserved, not replaced with native lower-ID ordering. |
| sparse MoE (903) | Shared expert plus gated shared branch and routed expert branch. Existing Kadan expert adapter accumulates selected experts in ascending ID order with BF16 rounding. |
| RMSNorm (925) | FP32 variance/normalization and `(1 + weight)` multiplication; final BF16. |
| TextModel (1330) | Fresh DynamicCache; position 0, one-token mask, layer loop and final norm. |
| ForCausalLM (1841) | BF16 LM-head output; no FP32 upcast on the no-loss path. Artifact FP32 values only contain the BF16 results. |

Execution fixes intra-op threads to 2, inter-op to 1, float32 matmul precision
`highest`, deterministic algorithms on and MKLDNN off; diagnostics verify and
record the effective settings. Parameters are CPU BF16 except A_log/dt_bias FP32.
The cache must have sequence length 1: convolution state BF16, linear recurrent
state FP32, full-attention K/V BF16. Hooks observe router IDs, probabilities,
boundary ties, and decoder-output hashes without extra forwards.

The independent expected values come from existing stdlib `stack_golden.py` and
`model_equations.py`, recomputed for one prefix with fixture dequantization and
per-layer scales. They do not consume HF or native outputs. Native incremental
linear recurrence and HF chunk equations are distinct paths; matching these two
fixtures is not proof of equivalence at other shapes, prefixes or weights.

Reference-only layer hashes cannot localize native divergence without comparable
native intermediates. This test contains no native GPU measurement and no
throughput claim. It does not establish actual-model correctness or support any
other modality. Synthetic path/dimension/file bounds are fail-closed guardrails
for trusted generated fixtures, not a sandbox for hostile checkpoint directories.
