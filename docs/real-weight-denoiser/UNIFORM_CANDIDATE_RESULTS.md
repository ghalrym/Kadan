# Decision: reject controlled-accumulation candidate v1 for production

Candidate `fp32-partial512-fp64-outputsum-v1` fixes development block 7 but fails
predeclared blocks 15 and 31 under unchanged final-output acceptance thresholds.
It also adds about 40–43% to measured serial short-block latency. Do not merge it
into production arithmetic, promote the two-GPU path, or infer BF16 readiness.
Keep the review-only implementation and evidence as a concrete rejected candidate.

## Frozen, causal implementation

Commit `f81b037fee04ed1cea62525fe4c653c5c119ddb2` froze one algorithm for both
attention-output and MLP-down projections: prepacked K=512 FP32 partial GEMMs,
FP64 accumulation of their output matrices, then one FP32 cast. All operations
use local rows and each path's current operands. Correcting attention output
causally recomputes normalization, MLP product and down-projection. There are no
block/coordinate-specific branches, full-row duplication, FP64 GEMMs in the
candidate, or combinations of saved substitution tensors. FP64 outer summation
cannot recover rounding already inside a partial GEMM.

The control evaluates both same projections in FP64 on current operands and
rounds once per projection, leaving the rest of the block unchanged. Acceptance
checks are final candidate output versus original pinned eager and versus the
independent FP64 oracle, both at `atol=rtol=2e-5`, with zero violations. Recorded
intermediate discrepancies are diagnostics, **not additional acceptance gates**.

## Development block and measured overhead

Both the causal two-projection FP64 control and frozen candidate pass both final
checks on block 7, for default and math SDPA, at eager and virtual-rank geometry.
Original baseline violations remain three/default and two/math versus eager,
and one per backend versus the oracle.

| GPU0 serial virtual-rank short block | Default SDPA median | Math SDPA median |
| --- | ---: | ---: |
| Original FP32 | 3.937 ms | 4.104 ms |
| Frozen candidate | 5.499 ms (+39.7%) | 5.858 ms (+42.7%) |
| Two-projection FP64 control | 31.709 ms | 31.685 ms |

Isolated identical local-row inputs give attention output projection
0.164→0.405 ms (2.47x), and MLP down-projection 0.394→1.127 ms (2.86x) under the
default backend. Packing takes 17.795 ms and adds 268,435,456 bytes (256 MiB) for
the two FP32 weights in this one block. This is not a full-model residency
measurement; packing all blocks would multiply that extra storage.

Timings use two warmups and five synchronized wall-time samples per operation;
all samples are retained. Comparison/CPU-copy audits are outside timings. Complete
block timing includes both virtual ranks running serially and diagnostic trace
bookkeeping equally in baseline and candidate, but no NCCL or concurrent two-GPU
execution. It is neither distributed speedup nor full-length/BF16/image latency.
No routine image rebuild occurred.

## Predeclared held-out captured blocks

Commit `1e95715` selected blocks 0, 15 and 31 before their evaluation; the candidate
remained byte-for-byte frozen. These are held-out blocks from the same prompt and
first cached timestep, not new prompts or scheduler trajectories. Each uses
128 target rows with the same captured prefix/RoPE/constants, independently
computed CPU FP64 reference and both attention backends. Capture and oracle SHA,
source hashes and candidate identity bind the reports.

| Block / backend | Candidate vs original eager violations | Candidate vs oracle violations | Original eager vs oracle violations |
| --- | ---: | ---: | ---: |
| 0 / default | 0 | 0 | 0 |
| 0 / math | 0 | 0 | 0 |
| 15 / default | 0 | **3** | 2 |
| 15 / math | 0 | **1** | 1 |
| 31 / default | **24** | **1032** | 1076 |
| 31 / math | **24** | **85** | 98 |

Block 31 candidate maximum error/bound ratios are 1.32468/default and 1.61126/math
against eager, and 5.53137/default and 2.00378/math against the oracle. Baseline
split has 22/default and 24/math eager violations, so this candidate does not
uniformly improve even eager compatibility.

The initial holdout run used the manual full-geometry reconstruction as its eager
baseline. A follow-up evaluates the actual pinned eager module, requires bit-exact
agreement with that reconstruction for every case, and repeats the comparisons.
All six baseline assertions pass and the complete result JSON is identical.
The first CPU-oracle launch had a missing PYTHONPATH and stopped at import; the
corrected invocation loaded the same predeclared inputs without source changes.

Original eager itself exceeds the independent FP64 criterion on later blocks.
That demonstrates a broader numerical limitation of the tested FP32 computation
at these shortened captured inputs; it does not excuse the candidate's failures,
justify widening tolerance, or establish a bug in distributed transport. This
experiment cannot identify every remaining contributing operation.

## Engineering disposition

Reject v1 as a general correction. Do not tune chunk sizes against these now-seen
holdouts and then call them unseen evidence. A new candidate would need a new
version and fresh validation cases. The next justified engineering work is to
localize the later-block eager/oracle discrepancy and determine whether a broader
controlled arithmetic strategy is necessary; the measured multi-GEMM prototype
already adds material overhead and should not be treated as an optimized kernel.

Actual BF16 accumulation changes remain a separate question requiring BF16
same-input evidence, fixed full-length chain/trajectory/image gates and measured
latency. No such run or production integration is justified by this result.
The original replay ladder remains stopped and both draft MRs remain review-only.

## Validation and preservation

All 19 CPU harness tests passed, including candidate tail handling, row ownership,
explicit BF16 rejection, nonzero/nonuniform scalar oracle and trace tampering.
The GPU0 development and holdout containers exited 0 under the 150-second host
limit, 4-GiB allocator limit, and existing thermal/physical guards. Final pinned-
eager holdout execution was 2026-10-08 22:44:24.079584956–22:44:44.221140953 UTC.
The exact live GPU1 API container and StartedAt remained unchanged and healthy;
no API pause, data writes to Postgres, or unrelated service changes occurred.
`evidence-uniform/` retains results, every timing sample, identity and resource
records; large FP64 tensors remain local.
