# Native Qwen3-TTS text projection

This original CPU component evaluates the pinned Qwen3-TTS `talker.text_projection`: Linear(2048,2048), SiLU, Linear(2048,2048), with both biases. It accepts one caller-owned text embedding and returns one projected embedding. It does not tokenize text, look up embeddings, execute the talker transformer or code predictor, sample codec tokens, decode waveforms, or synthesize speech. Full TTS remains unavailable in this native component; no queue/API/backend registration changes.

The source equation is `Qwen3TTSTalkerResizeMLP.forward` from pinned Qwen3-TTS dependency revision `022e286b98fbec7e1e916cb940cdf532cd9f488e`. The selected official CustomVoice 1.7B checkpoint revision is `0c0e3051f131929182e2c023b9537f8b1c68adfe`; config has hidden_act=silu and text/hidden size 2048. Four BF16 tensors are shape/dtype checked, read through bounded Shard reads and converted to FP32. Arithmetic uses scalar FP32 accumulation with contraction disabled and a stable SiLU evaluation.

Resources admits 33,570,816 bytes of retained weights, a 4 MiB metadata allowance, 4096-byte transfer staging and 16,384 bytes of execution scratch. These account for buffers, not whole-process RSS. Caller admits input and retained returned output. The CLI reserves 16,384 bytes for both heap buffers through output flush, records that publication accounting, then frees both buffers before releasing admission and reporting zero retained bytes. Serialized executor ownership is required; only cancellation is cross-thread. Weights persist across calls; pins prevent eviction during execution. Chunk/row cancellation checks, RAII scratch/pins and physical heap release before ledger release cover failure and unload. No new GPU allocator, WeightBacking policy or disk tier is introduced.

## Validation

GNU13 Debug CUDA OFF: six focused CTests pass (TTS, decision, video, checkpoint, resources, FIFO queue). Tests include insufficient load/execution admission, cancellation, pinned eviction, callback failure, bad dtype/nonfinite data, repeated calls and cleanup. Sparse BF16 diagonal weights give an independent closed-form Linear/SiLU/Linear oracle without Torch in CTest.

```sh
cmake -S native -B /tmp/kadan-tts -DKADAN_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/kadan-tts --target tts-tests kadan-tts-component -j2
ctest --test-dir /tmp/kadan-tts -R '^tts-component$' --output-on-failure
python3 native/tests/tts_checkpoint.py /tmp/kadan-tts/kadan-tts-component /path/to/model.safetensors
```

Real checkpoint: 3,833,402,552 bytes; header SHA256 `0e3594fc3c6c3fa05e2175fc7063bedbf9f55be7e9771f11e4770228cf9b2c26`. The explicit verifier reads only four tensors and records their hashes. It compares three inputs (zero, ramp, seeded) against independently assembled one-thread CPU FP32 Torch Linear/SiLU operations: 6144 output values, maximum absolute error `2.86102294921875e-6`, acceptance atol/rtol 1e-4. All runs report zero retained bytes, no GPU execution and no full TTS generation. This establishes component arithmetic, not BF16 end-to-end audio parity or speech quality.

Further TTS work needs token/embedding semantics, talker/code-predictor execution and cache ownership, and codec decoding before scheduler/API integration can expose native speech synthesis. STT remains the next separate modality track after this component milestone; it is not implemented here.
