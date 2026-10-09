# H3 block 0 attention and output projection component

`kadan-video-attention-component VAE_ROOT SHARD QKV_F32LE OUTPUT_COMPONENT`
consumes 1–8 supplied post-Q/K-normalization, post-RoPE tokens in
`[tokens,32,3,64]` per-head Q/K/V order. It performs non-causal scaled dot-product
attention over that complete supplied sequence, concatenates the 32 heads, then
applies block 0's learned 2048-to-2048 output projection and bias. It does not
append register/zero tokens, construct tiles, add a residual or generate frames.

## Pinned source and exact boundary

The installed H3 bundle is pinned by full-file SHA-256:

| File | SHA-256 |
| --- | --- |
| attention.py | a2d06ed14251937f98c8e903fb653282236222cc938569a37a1a3bb17b4579b0 |
| vae_vit.py | 1e11a02564f2acbcdaed991a9fcb2e7060815b39c7cc4c56ed1bbcddbddb172d |

`Attention.forward` splits interleaved per-head QKV, applies Q/K normalization
and rotary positions, calls attention, reshapes head-major results and applies
`to_out`. Its CPU `_sdpa_attention` calls PyTorch SDPA with zero dropout, no mask,
no explicit scale and no causal flag. Thus the scale is `1/sqrt(64) = 1/8` and
every query sees every supplied key, including later positions. The CUDA backend
also explicitly selects `causal=False`; no CUDA implementation is added here.

The decoder rejects `t_causal=True`. Its `apply_mask_preprocess` changes tokens
and coordinates before transformer execution; this is not an attention-score
mask. This component accepts no mask argument and does not claim to implement
that preprocessing or tile/suffix assembly. A small sequence is a numerical
probe, not a crop equivalent to full-sequence decoder output.

Only `decoder.transformer_blocks.0.attn.to_out.weight` [2048,2048] and `.bias`
[2048] are loaded from the F16 checkpoint. Matrix layout is [output,input]. The
checkpoint header SHA-256 is
`7bd6afbb1b2cfbf7c19901869eb224ec1b3e2a8cdbd8516d18f515c60f157c26`.
The selected weight/bias payload hashes are respectively
`539c8791759cb75985252c248720721325b3e0394d48de1056c35a305e1168b7` and
`edf9574eb4f0d847ee7f31fc69e631e6bd69d0b54908c065063237393f49f788`.

Native code implements stable max-subtracted softmax, weighted V sums and the
projection directly in F32 with contraction disabled. Finite-input/output checks
reject overflow. These F32 equations do not certify production FP16 rounding.

## Memory, cancellation and artifacts

Existing Resources admission precedes allocations; execution pins the resident,
and unload frees physical RAM before releasing the ledger entry. Selected
checkpoint reads use the existing bounded Shard reader, with a 4 MiB metadata
reservation and 4096-byte transfer staging. There is no GPU allocation.

- Resident converted weights: 16,785,408 bytes.
- Execution scratch: 16,416 bytes (one attended row, projected row and eight scores).
- Maximum caller-admitted input: 196,608 bytes.
- Maximum numeric execution total: 16,998,432 bytes.
- Maximum charged load total including input/metadata/staging: 21,180,416 bytes.

The CLI uses a 24 MiB host budget. The owner serializes load/execute/unload;
atomic cancellation is the sole cross-thread operation. Cancellation checks
occur during every checkpoint chunk, before resident publication, during input
validation, at every attention head, every 64 projection rows, after observation
hooks and before artifact publication. Hooks must not reenter execution. Pins
prevent eviction during execution, including observation callbacks.

Output is diagnostic F32 with header `KADAN_H3_ATTENTION_V1`, `TOKENS 2048`,
`F32LE`, each newline terminated, then at most 65,536 payload bytes. Unique
sibling temporary files and atomic no-overwrite publication retain prior
artifacts. Exceptions/cancellation remove partial files and release scratch/pins.
The trusted-directory owner remains responsible for aggregate artifact quotas.

## CPU validation

Default stdlib-only CTest checks 25 scalar reference cases: all lengths 1–8,
mixed queries, uniform softmax and sharply peaked scores targeting a later key.
Sparse projection rows mix distinct heads and include bias. Lifecycle tests
cover load/execution admission denial, wrong dtype/shape/nonfinite weights,
nonfinite Q/K/V and overflow, exact accounting, pinned eviction rejection,
pre-cancellation, first/final projection-group cancellation, hook exceptions,
overwrite rejection, temporary cleanup and destruction.

Optional real-checkpoint reference:

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
  native/tests/video_attention_reference.py COMPONENT H3_SOURCE_DIRECTORY \
  CHECKPOINT RETAINED_PR131_ROPE_EVIDENCE NEW_OUTPUT_DIRECTORY
```

It checks source hashes before extracting only `_sdpa_attention` by AST, without
importing SGLang. The retained RoPE report is pinned to
`186ab0edcdeb98088dc4141c7932578249fa069142837bb2ec1736ce129e0d77`;
each input payload is hash-checked. Actual selected F16 weights are converted to
F32 and used with Torch CPU linear. All 75 cases (712,704 values) passed fixed
atol=rtol=2e-4; maximum absolute error was 3.4570694e-6 and maximum tolerance
ratio 0.007835. Torch and native children were restricted to one CPU, with Torch
intra/inter-op pools each one. No GPU, full decoder or generation test ran.

## Remaining path to end-to-end native video

1. Finish block 0: learned attention residual scaling/add, norm2, gated SiLU
   feed-forward projections and feed-forward residual; validate against pinned
   source and real checkpoint at each boundary.
2. Assemble the actual latent grid, learned register tokens and zero suffix,
   corresponding 3D positions and supported mask preprocessing/postprocessing.
   Generalize execution to complete tile sequences with explicit bounded
   attention workspace; 1–8-token probes cannot stand in for full attention.
3. Execute all 36 decoder blocks with layer-indexed weights and bounded residency,
   then final normalization/output projection, suffix removal, 3D unpacking and
   VAE decode tiling/output handling. Establish complete CPU decoder references
   before any separately authorized GPU implementation/parity work.
4. Full text-to-video additionally requires native conditioning/text encoding,
   H3 denoising transformer (including the installed INT8/Turbo path), scheduler,
   latent preparation and frame encoding/publication.
5. Only then integrate a real video executor with queue admission, cancellation,
   residency switching and artifact ownership; keep API/orchestration/UI changes
   in separate reviews and test end-to-end output/quality.

This component is not registered as video generation. Current H3 generation
continues to use Python/SGLang. Shared queue/protocol and production Python/API
source are unchanged by this native patch.
