# Proposed arithmetic and trajectory gates — review required

## Recommendation

Keep the existing eager comparison and its failure as a compatibility gate. Add
an independent, serial emulation of rank-local arithmetic as a second reference
for parallel implementation correctness. Do not change production GEMMs yet:
the evidence identifies shape-dependent rounding, not an incorrect formula that
a new kernel would repair. No gate below is implemented or accepted by this
proposal, and the existing launcher must continue to stop at block 7.

The serial reference performs each rank's local GEMMs at the actual local row
count, then reconstructs global Q/K/V and evaluates its owned attention heads,
then reconstructs local output rows. This uses the same total useful GEMM work as
the parallel path, without full-row padding. Serial execution is an offline test
reference, not a replacement execution strategy or speed claim.

The reviewer-required separation is now explicit in two standalone review-only
entrypoints. `verify_parallel_geometry.py` compares distributed output against
virtual ranks at identical row/head geometry; it has not been executed and never
advances the existing replay ladder. `precision_oracle.py` independently evaluates
the full short-block equations in CPU FP64, including real-pair RoPE and explicit
softmax, without production or Diffusers operators. Both eager and virtual outputs
are assessed against that oracle at the unchanged `2e-5` bound. Passing the former
cannot substitute for passing the latter. See `ORACLE_RESULTS.md` for the observed
failure of the virtual path against the independent oracle.

## What the higher-precision evidence establishes

At [0,39,714], the split-minus-eager FP64 dot of the two saved MLP products is
-1.86123043216e-5. Their GPU-dot rounding errors differ by -8.09057653761e-6.
Together these explain the approximately -2.67028808594e-5 difference in MLP
outputs before multiplication by the common -0.9962648749351501 gate. Cancellation
with the common -2.483762741088867 residual leaves an output near 0.0551. The final
absolute discrepancy exceeds its 2.11023044586e-5 compatibility bound.

This is a decomposition using two different saved products, not a proof that
one path is accurate enough or that all upstream drift is harmless. The next
isolated check must evaluate both GEMM shapes on the *same* saved input products,
compare each to a CPU FP64 dot, and separately report input drift. Repeat for QKV,
attention output projection, MLP gate/projection and output at failing rows,
maximum normalized coordinates, and a fixed sample selected before execution.
Use FP64 sums of absolute products to report cancellation/conditioning alongside
absolute errors. A standard accumulation-error bound can be a diagnostic only:
backend-specific reduction details and overly loose worst-case bounds prevent
using it alone as a model-quality acceptance rule.

## Gate A: mathematical implementation and transport

1. Preserve bit-exact exchange round trips, prefix ownership, global-position
   RoPE, key-mask and padding checks independently of floating-point arithmetic.
   Include odd sequence lengths and masked keys, not only the even all-valid case.
2. Build the serial reference independently of `cached_block`, `sequence_to_heads`
   and `heads_to_sequence`: ordinary indexing/concatenation expresses ownership.
   Reusing `manual()` or the production permutation helpers as the sole oracle
   would permit the same bug in both paths. The pinned eager block supplies the
   mathematical operation definitions; intentional pinned float32 RoPE conversion
   remains explicit rather than pretending `block.double()` is a true FP64 block.
3. Match local GEMM dimensions, dtype, strides/contiguity, attention backend,
   scaling, masks, and precision flags. Compare stage outputs on identical inputs
   before comparing complete blocks. With these matched settings on the two
   identical cards, first require bit-exact results; any mismatch stops for
   localization, rather than introducing a tolerance from the observed maximum.
   A backend that cannot meet this requires a separately reviewed arithmetic
   bound supported by same-input higher-precision tests.
4. Add negative controls: permuted rank/head order, duplicated prefix, wrong RoPE
   offset and a changed mask bit must be detected. These checks establish that a
   shape-matched reference can reject wrong parallel math, not just match it.
5. Keep the original full-row eager FP32 report beside this new result. Passing A
   can establish implementation equivalence to the serial partitioned algorithm;
   it cannot erase the eager failure or authorize production deployment.

## Gate B: actual BF16 denoiser fidelity

Use the unchanged pinned BF16 eager pipeline as the user-visible baseline; never
replace this baseline with the partitioned reference. Evaluate the real 16384-row
shape and existing checkpoint, prefill, conditioning, prefix, scheduler and seed.
Keep the existing combined `atol=rtol=.02` criterion and report every violation.
This proposal does not increase it, add a permitted-outlier fraction, or relabel
FP32 failure as a pass. A separate, explicitly reviewed diagnostic run would be
needed to collect BF16 evidence despite the stopped FP32 compatibility gate.

Proceed in separate stages, stopping and retaining output at every failed gate:

- Teacher-forced blocks at fixed early/middle/late timesteps on identical eager
  inputs isolate local BF16 differences. Record all 32 block outputs, adaptive
  norm, and denoiser prediction, including per-rank and global metrics.
- Free-running 1/4/32-block chains use their own prior outputs. Check every layer
  and the final denoiser prediction against the eager chain; never reset from the
  teacher trace. A teacher-forced pass alone says nothing about accumulated drift.
- Paired full scheduler trajectories use identical initial latent and request
  conditioning. At each step compare denoiser prediction, scheduler update, and
  next latent. Prefix caches must follow the pinned pipeline lifecycle. Extend to
  fixed held-out seeds/prompts and supported mask/aspect-ratio cases before any
  production claim. The current single prompt and timestep cannot certify these.
- Decode final latents with the same VAE and compare final tensors/pixels plus
  task fidelity. Do not require identical encoded PNG hashes as a substitute for
  numerical comparison, or use visual similarity to excuse earlier failures.

Besides pointwise violations, record global relative L2, RMS error, maximum
normalized error, cosine similarity, error quantiles, and stepwise drift. For the
scheduler update also report error relative to the eager update norm, with a
predeclared absolute floor for near-zero updates. These expose coherent small
bias and late-step amplification that a pointwise bound alone may hide. They are
initially diagnostic: additional acceptance ceilings, update floors, pixel/task
criteria and the held-out request set must be fixed before inspecting candidate
results, using independent eager repeatability/precision controls and an explicit
quality requirement. Do not fit limits to this candidate's worst result.

There is no justified trajectory error budget in the present evidence. Until
those criteria are agreed and measured, report only limited numerical results,
not trajectory equivalence or production readiness. An eager-versus-eager repeat
and an eager FP32/BF16 precision control provide context, not automatic permission
for the candidate to be as inaccurate as either control.

## Efficient correction, if fidelity fails

The least invasive candidate is selective higher-precision arithmetic at the
first *measured BF16* divergence: for example FP32 accumulation through a sensitive
projection/gated residual, casting back at an explicit boundary. BF16 GEMMs may
already accumulate internally in FP32; merely requesting FP32 accumulation is
not guaranteed to change behavior. Saved-operand tests must establish which
rounding boundary matters before implementation. Such a change alters arithmetic
and requires its own quality and latency comparison; the FP32 block-7 result alone
does not justify promoting it.

An alternative is a fixed local GEMM tile/reduction algorithm whose arithmetic is
independent of rank count. This avoids redundant full-row work but requires kernel
engineering and benchmarking; ordinary deterministic flags do not promise
shape-invariant arithmetic. Neither alternative is an approved fix today.

## Proposed small review sequence

1. Add independent serial partition reference, same-input FP64 stage checks and
   negative controls, leaving the current compatibility gate untouched.
2. Review an explicit diagnostic-only BF16 trajectory protocol and fixed quality
   criteria, then collect bounded real-weight evidence. No full trajectory runs
   or API pauses are authorized merely by this document.
3. If needed, implement a localized arithmetic correction and repeat the fixed
   gates on held-out cases. Benchmark only after correctness and fidelity pass.
   Keep API integration and deployment outside these review-only changes.
