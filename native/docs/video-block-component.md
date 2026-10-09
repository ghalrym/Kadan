# Bounded native H3 decoder block 0

`H3DecoderBlock` composes pre-attention RMSNorm/QKV, Q/K RMSNorm and 3D NeoX RoPE,
noncausal attention/output projection, and the two scaled residuals around norm2 /
gated SiLU feed-forward. Input is one or two caller-supplied F32 hidden tokens of
width2048 and normalized [time,height,width] coordinates in[-1,1]. Output is one
`KADAN_H3_BLOCK_V1` F32LE diagnostic tensor. This is a real block execution, not
video generation, token assembly, the full36-block decoder, or a registered backend.

The existing public component interfaces and equations remain. Their private
compute sinks allow bounded admitted memory between stages, with no intermediate
files or rereads. All learned weights load from a single checkpoint::Shard descriptor
with the existing no-symlink, bounded-read and mutation checks. Path replacement
between stages cannot mix files. F16 weights are explicitly converted to CPU F32;
this does not establish production FP16/GPU parity.

## Ownership and memory

The serialized caller owns Resources and input reservations through return. Load
retains all four stage residents for subsequent calls; unload releases physical
allocations before ledger entries. Partial load failures unwind every prior stage.
All residents remain pinned through computation and atomic no-overwrite publication.
Cancellation and throwing hooks discard partial output and release intermediates;
weights remain reusable until unload. Hooks must not reenter the block. Atomic
cancellation is the only supported cross-thread operation.

| Payload/accounted envelope | Bytes |
| --- | ---: |
| Retained weights/frequencies | 268,574,752 |
| Maximum caller input and coordinates | 16,408 |
| Two QKV buffers + projected attention | 114,688 |
| Largest stage scratch | 90,112 |
| Maximum load with metadata/staging/caller | 272,789,560 |
| Maximum execution with caller | 268,795,960 |

The CLI admits288 MiB host RAM and no GPU memory. As for existing components, these
are explicit tensor/parser/scratch ledger bytes, not a process RSS ceiling; code,
allocator bookkeeping, small objects and filesystem page cache are not represented
as model payload. Metadata is bounded4 MiB, transfer staging4096bytes; no checkpoint
mapping or full-file payload load. SSD spill and generic queue registration are not
introduced by this diagnostic.

```sh
cmake -S native -B /tmp/kadan-native -DKADAN_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build /tmp/kadan-native --target kadan-video-block-component video-block-tests -j2
ctest --test-dir /tmp/kadan-native -R '^video-block-component$' --output-on-failure
/tmp/kadan-native/kadan-video-block-component VAE_ROOT SHARD INPUT_F32LE COORDINATES_F32LE OUTPUT
```

## Verification

Default CTest uses sparse F16 fixture weights and an independent stdlib scalar
end-to-end oracle: eight one/two-token cases, including resident reuse, mixed, zero
and tiny inputs. Lifecycle tests cover partial-load rollback, denied load/execution,
cancellation at each stage and immediately before publication, hook failure,
nonfinite/invalid input, pins, overwrite rejection, temporary cleanup, destructor
cleanup and exact host/device accounting. Existing stage, checkpoint, Resources and
FIFO tests also pass8/8 targeted tests. The block lifecycle/scalar test also passes
with AddressSanitizer and UndefinedBehaviorSanitizer. No queue behavior changed.

The optional `tests/video_block_reference.py` runs the pinned installed H3
TransformerBlock.forward, Attention.forward and FeedForward._forward_impl methods,
plus original normalization/rotary/SDPA helpers. Accelerated paths are disabled;
Torch uses CPU F32 with one intra-op/inter-op thread and one CPU affinity. It consumes
only initial tokens/coordinates and real checkpoint weights, never native stage
outputs. Source-file and selected-weight hashes are recorded before comparison.

Actual local H3 FP16 VAE: six zero/ramp/random cases, one/two tokens, all three
coordinate axes,18,432 compared values, max absolute error5.960464477539063e-8;
all pass atol=rtol=2e-4, zero nonfinite values. Every CLI reports resident_bytes=0
after unload. Evidence lives in task-4/native-h3-block-reference-v2, including exact
inputs, coordinates, outputs, binary and reference hashes. This is CPU block parity
only; full decoded video, large grids, all decoder blocks and GPU execution remain
unimplemented/unvalidated.
