# H3 block 0 residual and feed-forward component

`kadan-video-ff-component VAE_ROOT SHARD RESIDUAL_F32LE ATTENTION_F32LE OUTPUT_COMPONENT`
executes the tail of decoder block 0 for **one or two tokens**. Both inputs are
caller-admitted `[tokens,2048]` F32 arrays: the hidden state entering block 0 and
the already projected attention result. It returns a diagnostic block-tail
output, not frames. It does not internally run QKV, RoPE or attention, assemble a
real tile, or register a generation executor.

## Pinned equations

Installed H3 `base_module.py` SHA-256:
`14661aa1ef85c345eb1c64139d5640e98eab2dc35cd8b6e75151b51181d032d6`.
`TransformerBlock.forward`, `_scaled_residual_add`, and `FeedForward._forward_impl`
establish this order for the installed RMSNorm/gated-SiLU checkpoint:

1. `s = residual + attention * scale1`, with a per-channel learned scale.
2. `n = s / sqrt(mean(s*s) + 1e-5) * norm2.weight`, across 2048 channels.
3. `p = linear(n, ff.w1.weight, ff.w1.bias)`, width 16384.
4. Split `p` into **first-half gate**, second-half value, each width 8192;
   `g = SiLU(gate) * value`.
5. `f = linear(g, ff.w2.weight, ff.w2.bias)`, width 2048.
6. `output = s + f * scale2`, with a per-channel learned scale.

The actual seven F16 checkpoint tensors under `decoder.transformer_blocks.0.`
are validated: `scale1` [2048], `norm2.weight` [2048], `ff.w1.weight` [16384,2048],
`ff.w1.bias` [16384], `ff.w2.weight` [2048,8192], `ff.w2.bias` [2048], `scale2`
[2048]. Linear matrices are [output,input]. Their payload hashes are recorded by
the optional reference test; the checkpoint header is pinned to
`7bd6afbb1b2cfbf7c19901869eb224ec1b3e2a8cdbd8516d18f515c60f157c26`.

The native implementation converts selected F16 weights to F32 and evaluates
these equations directly with contraction disabled. SiLU uses stable positive/
negative branches to avoid exponential overflow. Nonfinite input, weights,
intermediate arithmetic and output are rejected. CPU F32 validation is not a
production FP16, fused-kernel or GPU-parity claim.

## Admission and ownership

Existing Resources and bounded checkpoint Shard contracts are unchanged.
Admission precedes allocation; loading reads selected tensors in 4096-byte
chunks with a 4 MiB metadata reservation. Execution pins the resident; unloading
frees its physical allocation before releasing the ledger entry.

| Charge | Bytes |
| --- | ---: |
| Converted resident weights | 201,424,896 |
| Execution scratch | 90,112 |
| Maximum two caller inputs | 32,768 |
| Maximum charged load, including metadata/staging/inputs | 205,656,064 |
| Maximum numeric execution total | 201,547,776 |

The CLI uses a 224 MiB host capacity. Scratch comprises residual, normalized and
result rows plus gate/value vectors. Serialized ownership governs load/execute/
unload; only cancellation is cross-thread. Checks occur per checkpoint chunk,
before resident publication, at each token, every 64 rows of both projections,
before gating, after the last observation hook and before artifact publication.
Hooks observe cumulative projection rows and must not reenter. Eviction is
rejected while execution holds its pin.

Artifacts use unique sibling temporaries and atomic no-overwrite publication in
a trusted directory. Exceptions/cancellation release scratch and pins and remove
partials. Header lines are `KADAN_H3_FEED_FORWARD_V1`, `TOKENS 2048`, `F32LE`,
followed by at most 16,384 payload bytes. The owner supplies aggregate disk quotas.

## CPU validation

Default stdlib-only CTest includes seven independent sparse-matrix oracle cases
covering both token counts, zero/tiny/mixed inputs, positive/negative/saturating
gates, signed/zero residual scales, bias and cross-channel mixing. Lifecycle
checks include load/execution budget denial, exact accounting, malformed and
nonfinite checkpoint data, mismatched inputs, overflow, pin-protected eviction,
pre-cancellation, first/final projection cancellation, hook exceptions, overwrite
rejection, partial cleanup, unloading and destruction.

Optional selected-real-weight reference:

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
  native/tests/video_ff_reference.py COMPONENT H3_SOURCE_DIRECTORY CHECKPOINT \
  RETAINED_PR130_QKV_EVIDENCE RETAINED_PR132_ATTENTION_EVIDENCE NEW_DIRECTORY
```

It hash-checks source before extracting the CPU residual and feed-forward AST
functions, without importing SGLang. GPU fusion is replaced by its CPU `None`
fallback; the actual extracted feed-forward function runs on CPU. Torch supplies
RMSNorm and F32 linear arithmetic over the actual checkpoint weights. Inputs are
hash-verified retained pre-attention residuals and attention outputs, so this
checks the block tail following the previous boundaries. Six real-weight cases
(18,432 final values) passed fixed atol=rtol=2e-4: maximum absolute error
5.96046448e-8, maximum tolerance ratio 0.000190722. The scalar fixtures separately
exercise the gated equations with substantial residual scales. Both Torch thread
pools and CPU affinity are one; no GPU execution occurs.

## Remaining decoder path

This completes the separately callable numerical boundaries for a bounded block-0
probe; it is not an integrated full-block executor or a full decoder. Next:

1. Compose those boundaries into one admitted block executor and validate complete
   block traces, including intermediate tensors and cancellation between stages.
2. Assemble actual grid/register/zero-suffix tokens and 3D positions, preserve
   pinned inference no-op mask hooks, and reject unsupported training-mask modes.
   Generalize attention and scratch management to complete tile sequences; a
   two-token probe is not equivalent to full-sequence attention.
3. Generalize weight names and residency across all 36 decoder blocks, add final
   normalization/output projection, suffix removal, 3D unpacking, VAE tiling and
   output handling; validate a complete decoder before separately approved GPU work.
4. Full text-to-video still needs conditioning/text encoding, installed INT8/Turbo
   H3 denoiser, scheduler, latent preparation and frame encoding/publication.
5. Integrate a real video executor with the shared queue only when it exists;
   keep API/orchestration/UI changes separately reviewed.

Current end-to-end H3 generation remains Python/SGLang. No shared queue/protocol,
API, production Python or UI changes are included here.
