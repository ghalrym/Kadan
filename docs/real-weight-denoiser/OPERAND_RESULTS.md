# Same-operand localization: no accepted arithmetic fix

All results concern development block 7 only. Original eager-versus-split and
FP64-oracle failures remain preserved at unchanged `atol=rtol=2e-5`. No holdout or
BF16 trajectory has been evaluated, and no production execution changed.

## Identical operands, then causal substitution

`isolate_operands.py` computes each path's Q/K/V, attention output projection,
MLP gate/projection/down-projection, normalization and elementwise results in
CPU FP64 from that operation's own exact saved FP32 inputs. Each isolated
operation passes the same-input bound, despite the composed block failing the
full independent oracle. For example, split down-projection's largest normalized
same-input error is 0.77716/default and 0.70221/math; attention output projection
is 0.41870/default and 0.43682/math. Neither statement guarantees the composed
block is accurate enough: inherited operand error and local rounding accumulate.

Norm1/modulation are identical between eager and virtual ranks; their first
path differences are Q/K/V GEMMs. Against FP64, even the shared normalization has
small ordinary FP32 rounding. Thus there is no evidence of a unique first faulty
formula. To attribute downstream influence, `probe_corrections.py` replaces one
operation's result with its same-input FP64 value rounded once to FP32, then
replays the unchanged downstream FP32 equations. Every output coordinate is
checked, not only the original failing coordinate. These are interventions for
localization, not candidate production FP64 execution.

## Reviewer-requested down-projection and first residual checks

`narrow_readout.py` evaluates both eager and split paths' own saved MLP products
with FP64 down-projection, rounds once, and applies the original saved GPU gate
and unchanged FP32 residual. Both corrected paths have zero final-output oracle
violations in both backends. Default corrected split also matches eager within
the existing bound. Math corrected split still differs from original eager at
[0,39,3013]: actual -0.039456844329833984, reference -0.03948163986206055,
error 2.47955322265625e-5, bound 2.0789632797241212e-5. No new oracle output
violations occur. Original [0,39,714] and math [0,39,1771] oracle violations clear.

That does **not** establish a complete fix:

| Same-input substitution | Default residual1 oracle violations | Default MLP output oracle violations | Final oracle violations (default/math) |
| --- | ---: | ---: | ---: |
| None | 1 | 7 | 1 / 1 |
| Attention output projection | 0 | 6 | 1 / 0 |
| MLP down-projection | 1 | 4 | 0 / 0 |
| Final multiply/add only | 1 | 7 | 1 / 1 |

Attention output projection correction removes the first residual violation
(maximum normalized ratio falls from 1.00702 to 0.94480). Down-projection
correction leaves that earlier violation intact and still has four/default and
one/math MLP-output violations versus the full-block oracle. Its math MLP-output
maximum ratio is 2.03252, despite a passing final output. Final-output cancellation
must not be used to declare the internal arithmetic fixed. Correcting final
multiply/add alone does not solve the output failure.

## Narrow accumulation experiment, not a selected fix

A review-only candidate breaks one projection's K dimension into 512/1024/2048
chunks, computes FP32 partial GEMMs, accumulates their output matrices in FP64,
and casts once to FP32. This duplicates no target rows and performs no FP64 GEMM.
It adds launches, partial-output traffic and FP64 additions; it has **not** been
benchmarked and is not claimed efficient in measured wall time.

For down-projection, K=512 or 2048 clears default final-output oracle and eager
comparisons, but math retains one original-eager mismatch. All variants retain
upstream/full-oracle intermediate failures. K=1024 still fails default output;
changing attention output projection alone also gives mixed downstream results.
These choices were explored on the development failure and are not predeclared
holdout acceptance criteria. Do not select a chunk size just because this example
passes. Results support investigation of a better local accumulation kernel,
not promoting this multi-GEMM prototype or a blanket FP64/full-row-padding fix.

A BF16 arithmetic change needs BF16 same-input evidence identifying the relevant
rounding boundary, then separately reviewed full-length chains, trajectories,
images and measured latency. The present FP32 result supplies no such evidence.

## Provenance, tests and execution

The stronger oracle test uses nonzero gates, nonuniform matrices, two target rows,
one prefix, nontrivial RoPE and independent Python scalar equations through the
entire block. Trace-integrity tests reject a modified intermediate file. All 17 CPU harness
tests passed in the pinned image.
`trace_binding.py` verifies the capture manifest and block SHA, both intermediate
files, backend reports/settings, source identity and backend operators before
new saved-trace comparisons. The binding of historical traces is retrospective,
explicitly labeled as such; it is not a pre-run source attestation. Future
captures should write this identity when producing artifacts. Original oracle
reports predate the binding and remain unchanged as historical results.

CPU work used four CPUs, 8 GiB/no swap and 180-second limits. The final GPU0-only
substitution run exited 0 at 2026-10-08 22:35:44.110144472 UTC (started
22:35:23.604818525), under the existing 150-second/4-GiB GPU allocator envelope.
It verified the trace binding before execution. GPU1 API container and StartedAt
remained unchanged and healthy. Evidence JSON is under `evidence-operands/`;
large operands stay local. The original stopped ladder remains untouched.
