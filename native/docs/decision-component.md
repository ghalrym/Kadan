# Native Laya terminal heads

This original CPU C++ component evaluates the installed Laya terminal scorer and action MLP from caller-supplied post-head activations. It does not tokenize text, run ModernBERT or either head transformer, build confidence features, calibrate probabilities, select actions, or answer questions. Full decision inference remains unavailable and no queue/API/backend registration is added.

The pinned source is `laya.common.DecisionModel` in dependency revision `8a6e1328cce2460a0e5aa348ad465bb1b5821cd2`, already used by Kadan. Scorer: LayerNorm(1024, epsilon 1e-5), Linear(1024,1024), exact erf GELU, Linear(1024,1). Action head: Linear(1028,256), erf GELU, Linear(256,2). Both checkpoint head counts and dimensions are fixed and checked. The caller supplies one 1024-element marker activation and one 1028-element action input (1024 pooled values plus four already-computed features). Outputs are three raw logits, not user-facing answers.

Only ten F16 tensors are read through the existing bounded Shard reader, then converted to retained FP32 weights. Resources charges 5,266,444 bytes of weights, a 4 MiB metadata allowance, 4096-byte transfer staging and 9228-byte execution scratch. Caller admits input and returned output; the CLI holds 8220 bytes for both heap buffers through publication, then frees both before releasing admission and reporting zero retained bytes. These are buffer accounting limits, not whole-process RSS guarantees. No GPU, new disk cache or WeightBacking spill behavior is introduced. Consecutive calls reuse weights until explicit/destructor unload; physical heap release precedes ledger release.

One serialized owner is required. Only the atomic cancellation flag may change concurrently; checks occur between read chunks and output rows. A resident pin prevents eviction while executing. Failure/cancellation unwinds scratch and pins; unload is idempotent. The optional observation callback is for diagnostics/cancellation, cannot reenter the executor and must not destroy it.

## Validation

GNU 13.3 Debug, CUDA OFF: focused CTest 5/5 (decision, video, checkpoint, resources, generation queue). New tests cover absent/double load, malformed/nonfinite weights, insufficient load and execution admission, pinned eviction, cancellation, callback failure, invalid/nonfinite inputs, repeated execution, unload/destructor cleanup and scalar numerical equations. FIFO coverage is from existing queue tests; decision execution is not newly queued.

```sh
cmake -S native -B /tmp/kadan-decision -DKADAN_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/kadan-decision --target decision-tests kadan-decision-component -j2
ctest --test-dir /tmp/kadan-decision -R '^decision-component$' --output-on-failure
python3 native/tests/decision_checkpoint.py /tmp/kadan-decision/kadan-decision-component /path/to/model.safetensors
```

Real checkpoint: `convaiinnovations/laya`, revision `7b928d828b7b0e022f929d9bd2e44165aa270148`, 842,609,210 bytes. Header SHA256 `3a42d8e7a96d5aa22c5bc9475d7ff092e32a832ae7c6973688f66f7a84048c04`. Only selected head tensors are read, no whole-checkpoint copy. The explicit verifier records per-tensor/input hashes and compares against separately constructed CPU FP32 Torch LayerNorm/Linear/GELU modules with one thread. Zero, ramp and seeded inputs: maximum absolute error `2.9760711667270456e-7` (acceptance atol/rtol 1e-4). All runs report zero retained bytes and no full decision/GPU execution. Scalar accumulation differs from optimized Torch reduction order; bitwise Torch parity is not claimed.

Further native MRs should establish encoder/head-transformer semantics and reference parity, then confidence/calibration semantics, and finally scheduling/API integration. This component alone does not establish end-to-end decision accuracy.
