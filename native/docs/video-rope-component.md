# H3 Q/K normalization and 3D rotary component

`kadan-video-rope-component QKV_F32LE COORDINATES_F32LE OUTPUT_COMPONENT`
executes the next boundary after H3DecoderQkv. Input is 1–8 tokens in
`[tokens,32,3,64]` per-head Q/K/V order, plus `[tokens,3]` F32 coordinates
(time, height, width). Raw payloads exclude preceding diagnostic headers.
Coordinates must be finite and within [-1,1]. The caller supplies the actual
normalized token positions; the component does not infer a grid or append tokens.

Each Q and K head is normalized across all 64 channels using affine-free
RMSNorm with epsilon 1e-5. H3's rotary ratio 0.75 rotates the first 48 channels;
the remaining 16 retain their normalized values. There are eight frequencies
per axis: `1 / 100^(i/8)` for i=0..7. Angles are `2*pi*coordinate*frequency`,
concatenated in time/height/width order and repeated across the two 24-channel
halves. Rotation pairs channel i with i+24 (NeoX split-half, not adjacent pairs).
V is validated and preserved bitwise. All equations execute in F32 with
contraction disabled; this does not certify production FP16 rounding.

Output has newline-terminated header `KADAN_H3_QK_ROPE_V1`, dimensions
`TOKENS 32 3 64`, and `F32LE`, followed by the transformed QKV payload. Maximum
payload is 196,608 bytes. Output uses unique temporary siblings and atomic
no-overwrite publication in a trusted directory; all failure paths remove the
temporary. The owner remains responsible for aggregate artifact disk quotas.

## Ownership and admission

This stage has no learned weights. Its 8-value frequency table is a real
32-byte admitted resident. Loading reserves before allocation; unloading frees
physical memory before ledger release. Execution pins that resident and admits
960 bytes of scratch (one 192-value head and two 24-value trigonometric tables).
Caller-owned QKV and coordinates must remain admitted until execution returns.
The CLI admits both arrays under a 256 KiB host budget; maximum numeric memory
including input, resident and scratch is 197,696 bytes. There is no GPU allocation.

One serialized owner controls load/execute/unload. An atomic cancellation flag
is the cross-thread operation. Cancellation is checked before loading, before
resident publication, at each token/head, after the observation hook, before
writing each head and before final publication. The hook observes completed
heads and must not reenter execution. Pins reject eviction while the hook runs.

## Pinned equations and CPU validation

The installed H3 video VAE bundle is pinned by full-file SHA-256:

| File | SHA-256 |
| --- | --- |
| attention.py | a2d06ed14251937f98c8e903fb653282236222cc938569a37a1a3bb17b4579b0 |
| vit_utils.py | fa142186b313ab33d049225d49de44e9f09527203cb1166a2b1f7f02667a9f25 |
| base_module.py | 14661aa1ef85c345eb1c64139d5640e98eab2dc35cd8b6e75151b51181d032d6 |
| vae_vit.py | 1e11a02564f2acbcdaed991a9fcb2e7060815b39c7cc4c56ed1bbcddbddb172d |

The checkpoint header hash remains
`7bd6afbb1b2cfbf7c19901869eb224ec1b3e2a8cdbd8516d18f515c60f157c26`;
its decoder configuration specifies 32 heads, width 64, non-affine Q/K RMSNorm,
rotary ratio 0.75 and theta 100. Native code implements these equations directly.

The default stdlib-only CTest checks scalar equations, all 1–8-token bounds,
zero/tiny inputs, three-axis coordinates, unchanged V including signed zero,
invalid input/coordinates, admission failures, pin protection, first/final-head
cancellation, hook exceptions, overwrite rejection and cleanup/accounting.

The optional reference command is:

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
  native/tests/video_rope_reference.py COMPONENT H3_SOURCE_DIRECTORY \
  RETAINED_PR130_QKV_EVIDENCE NEW_OUTPUT_DIRECTORY
```

It verifies the source hashes before extracting only the relevant AST functions
and RotaryEmbeddingND class. It does not import SGLang or its GPU runtime.
The retained QKV report is pinned to
`dd5484edd7f6a23ff44691e114ae55a7942574ca2490115127acc460961e023d`,
and every QKV payload hash is checked. The source functions normalize Q/K and
apply rotary embeddings independently from the C++ implementation. Reference
and child process are pinned to one CPU, with both Torch thread pools set to one.
Fixed tolerance is atol=rtol=1e-5. The test covers each axis and a 2x2x2 grid,
including zero suffix-position probes. These probes do not assemble suffix data.

## Remaining work

This is not attention or video generation and is not registered with the shared
queue. The next numerical boundary is non-causal scaled dot-product attention
and its output projection. Even block 0 still needs residual scaling, norm2,
gated SiLU feed-forward and its residual. The remaining 35 decoder blocks,
learned register-token plus zero-token assembly, full grid/mask/tile handling,
final decoder normalization/projection, 3D unpacking and video VAE output handling
also remain. Full text-to-video additionally needs native conditioning/text
encoding, H3 denoising transformer and scheduler, latent preparation, and frame
encoding/publication. Current H3 generation continues to use Python/SGLang;
these bounded F32 CPU stages are not a full decoder or an FP16/GPU parity claim.
