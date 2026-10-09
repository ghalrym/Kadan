# H3 block 0 RMSNorm and QKV component

`kadan-video-qkv-component VAE_ROOT SHARD INPUT_F32LE OUTPUT_COMPONENT`
executes the next boundary after `H3DecoderInput`: block 0's affine RMSNorm
(epsilon 1e-5) and QKV linear projection. It accepts 1–8 token-major F32
vectors of width 2048. The caller can pass the preceding component's payload
without its diagnostic header. This is an original scalar C++ implementation
of the checkpoint equations, with no dependency on Python inference runners.

The selected F16 tensors are `decoder.transformer_blocks.0.norm1.weight`
[2048], `attn.to_qkv.weight` [6144,2048], and `attn.to_qkv.bias` [6144],
where both attention names have the same block prefix. They are converted to
F32. Execution and accumulation are F32 with contraction disabled. This
reference boundary deliberately does not reproduce production FP16 rounding.
Nonfinite weights/input/output and overflow in the RMS sum are rejected.

The output starts with `KADAN_H3_DECODER_QKV_V1`, a dimensions line
`TOKENS 32 3 64`, and `F32LE`, each newline terminated, followed by the F32
payload. For each head, the three 64-wide vectors are Q, K, then V. It is
incorrect to interpret the 6144 columns as three contiguous 2048-wide vectors.
No Q/K normalization, rotary embedding, attention, output projection, residual,
feed-forward, further blocks, or frame decoding runs. Full video generation
remains unsupported and this diagnostic is not registered with GenerationQueue.

## Resources and ownership

The existing Resources ledger admits 50,364,416 resident weight bytes before
allocation, 4 MiB bounded checkpoint metadata, and 4096 transfer staging bytes.
Execution pins the resident and charges 32,768 scratch bytes. Caller-owned
input (at most 65,536 bytes) must remain admitted through publication; the CLI
does this with a 64 MiB total host budget. No GPU reservation or allocation is
made. Eviction frees the physical weight allocation before releasing its ledger
entry. Execution, loading and eviction have one serialized owner; cancellation
is atomic and may be requested across threads.

Cancellation is checked during each 4096-byte load chunk, each token, every
64 projection rows, before each payload write, and before publication. The
row hook supports deterministic cancellation and observation, not concurrent
or reentrant execution. The output directory must be trusted. Unique temporary
siblings are removed on failure, and publication atomically rejects overwrite.
Each payload is at most 196,608 bytes; the owner must enforce aggregate disk
quotas across artifacts. The implementation does not treat disk as fast RAM.

## Validation

`video-qkv-component` CTest uses sparse checkpoint fixtures and independent
scalar RMSNorm/projection equations. It checks per-head ordering, 8-token
output, incorrect dtype/shape, nonfinite values, bounds, cancellation inside a
token, pinned eviction rejection, allocation denial, exact resource accounting,
exception cleanup, no-overwrite publication and destructor cleanup. Existing
video-input, checkpoint and Resources tests are also run.

The optional `tests/video_qkv_checkpoint.py COMPONENT CHECKPOINT NEW_DIRECTORY`
uses Torch CPU F32 RMSNorm and linear as a separate numeric reference. Set
`CUDA_VISIBLE_DEVICES=''`; it pins itself and its scalar child to one available
CPU and sets both Torch thread pools to one. Zero, ramp, and seeded random
inputs cover every token count 1–8. Fixed acceptance is atol=rtol=2e-4. Its
report records checkpoint header/selected tensor hashes, binary and reference
source hashes, input/output artifacts, and per-case error. This is not full
H3 parity, GPU validation, production FP16 parity, or video generation.
