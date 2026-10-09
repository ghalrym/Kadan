> Historical benchmark record: temperature readings and thresholds below describe the original machine-local procedure, not Kadan runtime policy. Current benchmark temperature monitoring belongs outside the repository; see [external monitoring](/docs/EXTERNAL-BENCHMARK-MONITORING.md).

# API validation at b57db7173d6735f93244c6300539ef421da9a9f6

Independent review and exact-head API/native CI cleared this execution. API PR run 37875916784 and native PR run 37875916654 passed (push runs 37875912775 and 37875912770 also passed).

## Isolated baselines passed

Each case ran once in a fresh eager process with the frozen 2048²/40-step BF16 settings. Raw VAE output was finite and all 40 transformer/scheduler pairs completed. The original API container ID/config and small-model readiness were restored after each case.

| Case | Model load | Isolated generation | Complete window including restoration | PNG bytes |
|---|---:|---:|---:|---:|
| A, apple / seed 42 | 7.403604 s | 276.723502 s | 375.663159 s | 5,602,120 |
| B, teapot / seed 314159 | 7.724432 s | 276.800584 s | 375.537363 s | 5,976,606 |

A SHA-256: `6df59d86af4143efd1e3e5b419812754bd2b9ed1b10ccd33c3bb2ecac4dbf42b` — byte-identical to the prior full-trajectory reference.

B SHA-256: `76f0d6b06a3d1550457f7f6e65ea9dd6bf32da7a89955036ddd65708976465be`.

These are isolated baseline timings, not normal HTTP API performance or a dual-GPU speed claim. Evidence lives in `api-baseline-reviewed-b57db71-{A,B}` under the task evidence root. No duplicate baseline runs were made.

## A → B → A stopped at the unchanged thermal guard

The temporary API reached text readiness with both explicit 22.5 GiB logical device budgets. FIFO submission order was A1, B, A2. The first image's text handoff took 0.010334 seconds. Cool admission retained its rejected 61.0 C reading before obtaining the required sub-60 C samples.

A1 returned HTTP 503 after 113.695872 seconds: `Image CPU thermal guard reached: 80.0 C`. No image was published. This is failed-request latency, not generation speed. The controller fenced the pair; the queue began cooling the next job before the validation procedure stopped the temporary API. Following queued requests were aborted by teardown/recovery, so this window does not establish successful A/B/A parity, persistent reuse, handoff performance, cancellation recovery or peer-failure recovery.

The rejected 80.0 C reading and bounded rank stderr were retained. The last external peak sample showed Tccd3 at 79.5 C while Tccd5 was 50.0 C. Both workers had been pinned to shared affinity CPUs 0/1, which are distinct cores within the same last-level cache/CCD domain. CPUs 0/8 belong to different domains. This supports investigating heat concentration; it does not prove a placement change solves the guard trip.

Exact original API restoration completed after 298.338088 seconds. Queue counts returned to zero. Evidence is `api-queue-reviewed-b57db71-aba`; summary is `mr125-b57db71-results.json`. All failures and prior evidence remain preserved. Refreshed retained-directory bytes: 33,642,895,011 against the unchanged 34,359,738,368-byte ceiling.

## Proposed placement correction, pending review and hardware validation

Keep both rank processes within the same maximum of two allowed CPUs, with unchanged one intra-op/one inter-op thread per rank. Prefer CPUs from different highest-level shared cache domains when Linux topology is available; preserve the deterministic first-two fallback when topology is unavailable or only one domain is allowed. On this host the CPU-only probe selects [0,8]. Log the chosen affinity. No thermal threshold, memory budget, execution deadline or cleanup deadline changes.

26 focused CPU lifecycle/transport tests pass, including multi-domain placement, lower-cache avoidance, restricted affinity, unavailable topology, GPU alias rejection and cleanup. GPU retry requires independent review and exact-head CI. The baseline runner, settings, model arithmetic and saved A/B images are unchanged; retain the existing baselines with their original b57db71 provenance when evaluating the separately reviewed placement-only application change.
