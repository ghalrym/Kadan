# H3 VAE decoder input component

This original CPU C++ slice loads six real F16 tensors from the installed H3 video VAE and evaluates, for each token, `z * latents_std + latents_mean`, the 24-by-24 `post_quant_conv`, then the 2048-by-24 `decoder.x_embedder`. Accumulation is scalar FP32 with contraction disabled. It emits a diagnostic tensor, not frames. This is not an upstream full-decoder parity certification or text-to-video implementation.

The existing video paths are Python/SGLang workers. GenerationQueue still accepts only text/image executors; this change adds no video queue/protocol/API registration. Keep full video generation unavailable until a complete executor exists. Next reviews should cover decoder block semantics and reference parity, subsequent decoder stages, and finally queue/API integration separately.

The executor uses Resources admission, pinning and physical cleanup before release. Its 207,392-byte decoded weight bank remains in RAM across calls. Checkpoint reads use the existing bounded Shard reader, not a copy of the 5.2 GB checkpoint. No GPU, mmap, WeightBacking spill policy, disk-as-RAM claim, or new residency scheduler is introduced. The shared reader gains additive F16 metadata/width and an explicit metadata-value string allowance: default 512 remains unchanged; this component opts into 4096 because the real checkpoint contains a 2013-character metadata value. Names remain bounded at 512.

Serialized owner only; the cancellation flag is the only cross-thread operation. Caller admits and owns token-major FP32 input. Maximum 4096 tokens bounds each diagnostic payload to 32 MiB plus a short header. The trusted output directory is caller-owned; aggregate retention/quota is caller responsibility. A sibling temporary file is removed on every failure/cancellation, with atomic no-overwrite publication on success. Cancellation is checked between checkpoint chunks and tokens and immediately before publication; cancellation concurrent with publication may complete successfully. Output disk I/O is synchronous.

RAM reservations: 4 MiB metadata allowance, 207,392 bytes resident weights, 4096 bytes checkpoint staging, 8384 bytes execution arrays, plus caller-owned input. These track model/data buffers, not allocator/runtime overhead or the whole process RSS. CLI admits its input and releases it on exceptions; it uses an 8 MiB ledger.

## CPU validation

```sh
cmake -S native -B /tmp/kadan-video -DCMAKE_BUILD_TYPE=Debug -DKADAN_ENABLE_CUDA=OFF
cmake --build /tmp/kadan-video --target video-tests kadan-video-component checkpoint-tests resource-tests generation-queue-tests -j2
ctest --test-dir /tmp/kadan-video -R '^(video-component|checkpoint|resources|generation-queue)$' --output-on-failure
python3 native/tests/video_checkpoint.py /tmp/kadan-video/kadan-video-component /path/to/minimax_h3_video_vae_fp16.safetensors /new/evidence/directory
```

GNU 13.3 Debug CPU-only: all four focused CTests pass. Tests cover cancellation, eviction while pinned, admission failure, ledger cleanup, repeated unload, malformed/nonfinite weights and inputs, FP16 edge values, output limit, overwrite refusal, callback errors and temporary cleanup. Existing queue tests retain FIFO/cancellation coverage; there is no new video queue execution.

Real checkpoint: 5,207,808,496 bytes; header SHA256 `7bd6afbb1b2cfbf7c19901869eb224ec1b3e2a8cdbd8516d18f515c60f157c26`. Zero, ramp and seeded inputs produce 6144 bitwise-equal FP32 values against separately written Python scalar equations. Output SHA256 `f8bfaa584dacc419264b9ae375bf3af1cf3d0cad50a0b7c9de4896628e7f75de`. CLI reports `resident_bytes=0`, `full_video_generation=false`, `gpu_execution=false`. The verifier records selected tensor hashes; it neither copies nor hashes the whole checkpoint. No GPU execution or live API source change was needed.
