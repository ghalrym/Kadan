> Historical benchmark record: temperature readings and thresholds below describe the original machine-local procedure, not Kadan runtime policy. Current benchmark temperature monitoring belongs outside the repository; see [external monitoring](/docs/EXTERNAL-BENCHMARK-MONITORING.md).

# Separate BF16 first-slice implementation — source review pending

Protocol v1 was independently cleared at `1e6e0c64f2db5675af1b5ff15d84bef9e011da64`
for implementation. This new source still requires independent review and
exact-head CI before execution. No BF16 GPU run, API pause or live deployment
change has occurred for this implementation.

The original `launch.py`, `replay.py`, failed reports and production
`api/inference/image/parallel.py` remain unchanged. Candidate v1 is not imported
or selected. Later steps 20/39 are not implemented or executed by this first-slice
entrypoint; every verdict keeps them pending and `fp32_protocol=failed`.

## Entry points and admission

- `launch_bf16.py` defaults to a read-only JSON plan. Execution requires
  `--execute-reviewed` with the exact clean checkout commit, `--capture` pointing
  at the existing verified capture, a fresh `--evidence` directory, and
  `--review-record` naming independent source review and bounded execution approval.
- Admission verifies the original capture manifest SHA and every packet, the
  unchanged Ulysses source SHA, and completed successful push runs for both
  `API tests` and `Native worker CPU tests` at that exact commit through GitHub.
- The review record must contain protocol `bf16-application-diagnostic-v1`, exact
  `source_commit`, `decision=approved-for-bounded-execution`, `reviewer`, and
  `review_reference`. Do not manufacture this record from protocol approval alone:
  it must point to actual independent implementation review and execution
  authorization. The launcher checks fields and identity, not the truth of an
  arbitrary operator-authored attestation.
- `supervisor_bf16.py` takes only `bf16-replay`, holds the existing inference-lock
  inode, launches two ranks with the static loopback rendezvous and reaps the
  owned process group on exit, cancellation or timeout.

The launcher retains the absolute 2100-second pause budget and 600-second recovery
reserve, physical/thermal/owner admission, 128-GiB no-swap host limit, 22-GiB GPU
allocator limits, exact-container restoration and native model readiness checks.
It mounts the capture read-only and counts capture plus new evidence against the
32-GiB active-artifact ceiling. It neither captures new weights nor touches prior
results. Hash verification and CI checks happen before stopping the API.

## Replay semantics

`replay_bf16.py` checks independent expected global-row/local-head ownership of
sequence-to-head exchange and independent inverse ownership, in addition to the
round trip. It then loads all 32 original trained BF16 blocks and request constants.

Teacher-forced eager evaluation runs on rank 0 and is broadcast separately from
input storage, preserving each rank's unchanged incoming tensor. Ulysses output
is compared to both eager replay and captured BF16 output. Eager itself must
reproduce capture. For each 1/4/32-block chain, reference and parallel hidden
states advance independently; neither resets to teacher hidden state. Each block
output is checked and the 32-block chain includes final adaptive norm/projection.

All numerical checks keep `atol=rtol=.02`, zero violations and zero nonfinite.
Results are persisted before rank-wide acceptance. Rank-0-only reference audits
avoid double-counting replicated references; sharded counts sum across ranks and
extrema use maxima. Global quantile intervals are derived from summed histogram
counts with identical edges, never averaged rank quantiles. The global worst
normalized point retains its rank, global coordinate, actual/reference and bound.
When any nonfinite value occurs, global error metrics and quantile intervals are
null; the separately labeled finite-only histogram still reports its finite count.
Reports include signed worst normalized coordinates/bounds,
absolute errors, RMSE, relative L2 and fixed-histogram quantile intervals. Up to 32
failing coordinates are listed with truncation marked; failed tensors are saved
before stopping. Intermediate operators do not introduce acceptance gates.

Reference and Ulysses attention operators are profiled once outside timing to
record selected dispatch without forcing a backend. Capture/source identities,
precision flags and the verdict history are retained. A first-slice pass cannot
clear either the prior FP32 failure or the pending overall protocol.

Cached-core timing is reached only after every numerical stage passes. Each rank gathers its remaining stage budget over the control group; both branch
on the same minimum. If that minimum is less than 120 seconds, timing is skipped
explicitly on both ranks. The budgets and agreed decision are persisted.
Otherwise it uses two warmups/five synchronized repeats, including sharding,
control consensus, communication/layout, all blocks, norm/projection and gather.
CPU audits remain outside the timed region and every repeat is checked. The
supervisor deadline still bounds execution; 120 seconds is admission headroom,
not a guarantee timing will finish. No image/trajectory speedup is claimed.

## CPU validation

Contracts cover default read-only behavior, exact source review/CI admission,
independent exchange ownership that rejects a canceling permutation, unchanged
thresholds/nonfinite rejection, rank aggregation, independent chain state,
retained prior/pending verdicts, deadline clipping/recovery reserve, and owned
process-group cleanup on timeout. These are CPU contracts, not evidence that
full-length CUDA replay or restoration has run successfully. Source review and
exact-head green CI remain execution prerequisites.
