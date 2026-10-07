# One-token native linear-attention sublayer

This is an executable, original **hidden input → residual output** path for one
batch-one linear-attention sublayer, with a bounded CPU sequence reference and a
single-device CUDA owner. It is stacked on #106. No actual checkpoint is opened
or loaded; supplied synthetic weights exercise the complete path. Full attention,
MoE, a complete decoder, generation, and production integration remain separate.

## Mathematical source and scope

The semantic reference is the verified installed Transformers 5.17.0 file,
byte-identical to upstream commit `856157a2f3e9594954310df18fdccc31ffddebe9`:
[Qwen3_5Moe model reference](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py).
Relevant sites are convolution 254–268, L2 norm 296–299, recurrence 440–499,
projection/grouping/gating 562–665, and offset norm 925–939. File SHA256 is
`42b6ca1cd9a2f0754c3dec97f0708dfa7e97e1b6edd9fac88f3139e4e36316c2`.
Only mathematical/configuration conventions were inspected. The scalar reference,
thread decomposition, kernels and integration here are original; no external
model/kernel implementation is copied, ported, wrapped or called.

The installed profile is hidden 2,048; key heads 16, value heads 32; key/value
widths 128; convolution length 4; epsilon 1e-6. The new explicit `linear::Config`
can describe those shapes and smaller synthetic ones. It does not discover model
roles or load files; a future reviewed binder must derive it from the manifest.
Weights comprise scalar-scaled FP8 QKV, z and output projections; BF16 input norm,
convolution, a/b projections, A_log, dt_bias and output norm. Unsupported shapes,
nonfinite weights and auxiliary floats not exactly representable as BF16 fail.

Per token:

1. Offset RMSNorm over hidden channels, then QKV, z, a and b projections.
2. Shift each raw-QKV convolution channel's oldest-to-newest BF16 history; append
   current QKV. Apply depthwise cross-correlation (weight 0 multiplies the oldest
   retained sample), then SiLU. Initial history is zero padded.
3. Split contiguous Q, K, V regions. Q/K L2 normalization uses **sum**, not mean,
   plus 1e-6; Q additionally divides by sqrt(key_dim). Value head `h` uses key head
   `h / (value_heads/key_heads)`, without physically copying Q/K.
4. Compute `beta = sigmoid(b)`, `g = -exp(A_log)*softplus(a+dt_bias)` and
   `decay = exp(g)`. Stable sigmoid/softplus avoid exponential overflow where
   the mathematical expression is finite; unrepresentable results fail closed.
5. Each value head owns FP32 `S[key_dim,value_dim]`. Apply:

   ```
   S_bar = decay * S
   prediction = transpose(k) * S_bar
   delta = beta * (v - prediction)
   S_new = S_bar + outer(k, delta)
   core = transpose(q) * S_new
   ```

6. Direct-weight RMSNorm over each value head, followed by SiLU(z), then output
   projection and addition to the original hidden input.

The owner does not include the post-attention MoE normalization/feed-forward
block. It has no RoPE or KV cache because those belong to the full-attention
family. This PR closes the linear path rather than pretending to be a decoder.

## Explicit numerical contract

`bf16_round` is round-to-nearest, ties-to-even, including signed zero and
subnormals. Nonfinite inputs and rounding overflow to infinity are errors.
Input/output API spans store exact BF16 values in float containers for use with
existing FP8 projections. Device auxiliary weights and convolution history are
actually packed 16-bit BF16; recurrent state and intermediate scratch are FP32.

| Boundary | Contract |
| --- | --- |
| Pre-norm | FP32 square reduction/normalization and `1+weight`; round result to BF16 |
| QKV/z and a/b | Round each projection result to BF16 |
| Convolution | FP32 products/sum; round to BF16 before SiLU; round activation to BF16 |
| Q/K L2 and scaled Q | FP32; no BF16 round after normalization |
| beta | Sigmoid evaluated from BF16 b; round beta to BF16 before FP32 recurrence |
| g/decay and state | FP32, with explicit finite checks |
| Recurrent output | Round to BF16 before direct gated normalization |
| Gated normalization | FP32 normalization; round normalized value to BF16, multiply BF16 weight and round again; FP32 SiLU(z) product, then BF16 |
| Output projection/residual | Round projection to BF16, add original BF16 residual in FP32, round final output to BF16 |

The CPU reference and CUDA core use explicit ordered FP32 sums/products for dense
auxiliary projections, convolution, normalization and state updates. CPU floating
contraction is disabled; CUDA state/dense products and additions use explicit RN
operations. The CPU FP8 projection oracle accumulates in double, while the existing
original CUDA projection uses FP32 FMA and a parallel reduction. Elementary math
libraries may also differ. Therefore this is a specified rounding contract with
bounded numerical comparisons, **not a claim of bitwise Transformers, PyTorch,
cuBLAS or actual-model parity**. No activation quantization/calibration is added.
The finite-range policy deliberately rejects pathological overflow rather than
propagating NaN state. Adversarial boundary values may require a separate numerical
policy review before production use.

Several early kernels are intentionally simple (serial row normalization, direct
small dense projections, one value-coordinate recurrence per CUDA thread). They
establish state/rounding correctness before optimization. No speed claim is made.

## Ownership, admission and progress

`Fp8Projection::matvec_device` takes disjoint aligned caller-admitted device spans.
It retains/pins immutable projection storage, executes on the legacy stream,
synchronizes and checks its status word before returning. It never stages
activations through host RAM or allocates per call. Caller pins borrowed buffers
until return; allocation provenance and current-device ownership are trusted
native caller obligations, as for the existing borrowed math primitives.

`cuda::LinearAttention` owns three FP8 projection owners, one `SequenceState` arena,
and one packed-auxiliary/scratch/status allocation. All five reservations use the
same supplied `Resources`; no separate imaginary capacity pool is created.
Auxiliary upload uses a fixed 2 KiB stack staging buffer. Host weights, fixed
control objects, stack/runtime/allocator overhead and CUDA context require caller
host/device headroom; they are not hidden in requested tensor byte accounting.

The CPU `Reference` borrows immutable weights and checks an explicit requested
state/workspace budget before allocating its history/state/scratch vectors. That
budget excludes borrowed weights and allocator/object overhead. The CUDA owner
uploads weights once; constructor inputs may be released afterward. Scratch and
state persist; `step_device` creates no model-sized buffers. Its scalar status
readbacks and synchronization are correctness boundaries, not a fully asynchronous
execution or performance claim.

Every forward starts one owner-bound state transaction and pins workspace/state.
The state is acknowledged only after the entire sublayer, including residual and
numeric checks, succeeds. Only then does commit advance the token count. Any
failure after begin invalidates the sequence, including errors after convolution
or partial recurrence mutation; reset must physically zero convolution/recurrent
storage before another token. No rollback or prefix reuse is claimed. Capacity
rejection occurs before begin and does not mutate the committed sequence.
Numerical/input failures can be followed by explicit reset; uncertain CUDA errors
poison the owner until close. Cleanup failure retains reservations and latches
against automatic retries. Keep the shared resource authority alive independently
of a failed owner so quarantined charges cannot disappear unnoticed.

The class is creating-thread/current-SM86-device only. It is blocking and has no
mid-step cancellation callback. A future decoder must coordinate cancellation and
commit across all sublayers/devices; this local one-layer commit is not a global
model transaction. Device input/output may exactly alias, but partial overlap is
rejected. Discard external output after any failure, even if some values were
already written. `read_state` is diagnostic and refuses invalid state.

## CPU evidence and staged GPU proposal

All twelve CPU CTest cases pass with ASan, LSan and nonrecovering UBSan; the complete
SM86 CUDA build/link passes with CUDA12/GCC12 and compile parallelism two. No new
GPU execution occurred. Tests cover BF16 ties, subnormals and overflow; exact budget
rejection; scalar closed-form multi-token recurrence with separate value heads;
nonzero older convolution taps; key/value-axis layout; direct state inspection;
reset/replay; invalid input; and an expected exponential failure after history has
already changed. The existing blocked Python fixture generator remains untouched.

The main independent scalar test has hidden size 2, two key heads/four value heads,
key/value width 1 and a two-tap `[0.5,1]` convolution. With zero a/b/A_log/dt_bias,
decay and beta are both 1/2, giving the independent closed form
`S' = .5*(1-.5*k*k)*S + .5*k*v`. For input sequence `[1,1]`, `[1,-1]`, `[-1,1]`,
the expected residual outputs are:

```
[ 1.734375,   1.734375   ]
[ 1.734375,  -0.73046875 ]
[-0.73046875, 1.734375   ]
```

The test also asserts independently calculated FP32 state values at each step,
not merely output finiteness. A second two-key-coordinate/two-value-coordinate
case detects transposed state axes and checks zero-key-coordinate state remains
zero. A nonzero a/b/A_log/dt_bias case independently checks beta .26953125 and
decay approximately .084056817, including accumulated two-token state. These tiny cases do not validate actual-model parameter values.

Only after independent exact-head review and explicit clearance, the opt-in
`kadan-linear-attention-parity --allow-gpu-validation --device 0 --stage N` can run
in three separately gated stages, stopping at any unexpected failure:

| Stage | Work | Kernel launches |
| --- | --- | ---: |
| 1 | One complete token, CPU output/state comparison, reset/zero and cleanup | 10 |
| 2 | Three-token evolution with nonzero learned decay/update gates, then reset/replay | 60 |
| 3 | Expected overflow after history mutation, invalid-step rejection, physical reset and cleanup | 8 |

Each process uses the same tiny shape above: **5,376 peak requested device bytes**
(three 1,280-byte projection slabs, 512 state bytes, 768 auxiliary/workspace/status
bytes, 256 caller IO bytes), under a 64 KiB device cap. Host fixture/reference
numeric buffers are below 16 KiB; context/runtime overhead is additional. Finite
outputs must match the designed BF16 goldens exactly, convolution bytes exactly,
and recurrence within absolute 2e-5. The harness checks zero-ledger cleanup and
never registers as a CTest. Unexpected failures terminate the dedicated process
without automatic reruns, resets of the GPU, or service/power changes. None of
these stages loads a checkpoint, generates text or benchmarks throughput.

After review and validation, the next coherent milestones are the full-attention
KV sublayer, MoE/shared-expert execution, and a decoder loop with global sequence
transactions and explicit device handoffs. Only then should a separately reviewed
adapter expose the native decoder to production orchestration.
