# Independent short-block oracle: split path remains failed

The reviewer-requested separation is implemented without changing `replay.py`:

- `verify_parallel_geometry.py` is a separate two-rank diagnostic that compares
  the distributed block with one-GPU virtual ranks using identical local row/head
  geometry. It records the original eager comparison separately and requires
  bit-exact virtual/distributed output. **This entrypoint is not executed.** It
  requires the existing reviewed two-GPU supervisor/resource/lock envelope; it
  does not authorize a live API pause.
- `precision_oracle.py` implements the complete shortened block equations in CPU
  FP64 independently of production/Diffusers block, normalization, RoPE and SDPA
  helpers. It compares both saved eager and virtual-rank outputs to this reference
  using the unchanged `atol=rtol=2e-5`. A match to virtual ranks cannot hide an
  oracle failure. Saved distributed output is not available for this CPU run and
  its verdict is explicitly null, not passed.

## Results from the saved real block 7

| FP32 path | FP64-reference violations | Maximum error / bound |
| --- | ---: | ---: |
| Default eager | 0 | 0.6443298293406873 |
| Default virtual ranks | **1** | **1.097668380591383** |
| Math-SDPA eager | 0 | 0.7818884066044316 |
| Math-SDPA virtual ranks | **1** | **1.0263114888552165** |

Default virtual ranks fail [0,39,714]: actual 0.05514192581176758, FP64 reference
0.05511876240170244, absolute error 2.316341006514122e-5, bound
2.110237524803405e-5. Math virtual ranks fail [0,39,1771]: actual
-0.1152312159538269, FP64 reference -0.11525410791590585, absolute error
2.2891962078941397e-5, bound 2.230508215831812e-5.

The original eager-versus-virtual failures remain three/default and two/math.
Neither a matched attention backend nor an independent reference makes the split
path pass. This evidence supports retaining the stop and investigating arithmetic
accuracy before BF16 trajectory progression, not relabeling shape-related error
as acceptable. It does not establish that BF16 will fail or that a particular
arithmetic correction will succeed.

## Oracle semantics and limits

Inputs, BF16 checkpoint weights and the already-generated BF16 prefix are promoted
exactly to FP64. RoPE uses the exact captured complex64 coefficients promoted to
FP64, with complex128 pair rotation and **no float32 roundtrip**. Protocol
`captured-short-block-fp64-v1` fixes captured constants, the 128-row block, explicit
FP64 equations and symmetric `atol=rtol=2e-5`. Review this version before holdout
blocks or prompts.
Normalization, modulation, projection, RMSNorm, attention scores/softmax/value
sum, SwiGLU and residuals are evaluated in FP64. Score storage is bounded to one
head at a time. No FP64 prefill, new RoPE frequencies, full-sequence FP64 run, or
whole-model mathematical exactness is claimed. This is a higher-precision
reference for the fixed captured short-block problem, not arbitrary-precision
truth or the original pinned finite-precision algorithm.

The oracle's own implementation needs source review. Analytic CPU tests verify
pair rotation retains sub-FP32 precision, attention mask/head ownership, zero-gate
identity through the full equations, and independent verdicts that retain eager
and oracle failures even when transport matches. These do not certify every
possible block configuration. The current oracle covers cached target-only rows
with at least one valid key; unsupported all-masked input raises explicitly.

Execution used the pinned image on CPU only, four CPUs, 8 GiB/no swap, no network
or GPU access, and a 180-second timeout (exit 0). The full harness suite passed
15/15 CPU tests afterward under a 120-second timeout. No API/container deployment
or GPU residency changed. `evidence-block7/fp64-oracle-report.json` contains the
capture SHA, stage comparisons, original failures and all oracle failures; the
FP64 tensor remains local. No acceptance threshold changed.

Next: independently review these equations and run same-input FP64 operator
checks to locate the remaining error before selecting a localized correction.
The separate transport test can then be run under its reviewed envelope. The
BF16 teacher-forced, free-running block and scheduler/pixel gates in
`REFERENCE_PROPOSAL.md` remain necessary and unexecuted.
