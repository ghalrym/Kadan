> Historical benchmark record: temperature readings and thresholds below describe the original machine-local procedure, not Kadan runtime policy. Current benchmark temperature monitoring belongs outside the repository; see [external monitoring](/docs/EXTERNAL-BENCHMARK-MONITORING.md).

# Full independent BF16 trajectory: v2 passed

Executed source: `195fc7440005c449a57d77f4f312971a3280d22f`, from `/home/andrew/Projects/self-hosting/Kadan`, protocol `full-independent-trajectory-v2-rgba`. The source remained clean and frozen through all three separately admitted windows. This result does not activate or merge production API integration, and the diagnostic Torch/Diffusers runner is not a native C++ image executor.

## Numerical verdict

The fresh eager reference, independent eager repeat, and dual-GPU candidate each completed the full 40-step 2048×2048 trajectory at seed 42 and produced a finite decoded RGBA image. The repeat passed all 83 comparisons. Each candidate rank passed all 83 comparisons (166 total). Every comparison was bit-exact: maximum absolute error, violations and nonfinite counts were all zero, including float RGBA and uint8 pixels. RGB and alpha independently passed the original mean-error thresholds. Both candidate ranks used their own full pipelines and scheduler state; references were used only for audits.

All three PNGs share SHA-256 `6df59d86af4143efd1e3e5b419812754bd2b9ed1b10ccd33c3bb2ecac4dbf42b`. This validates one fixed prompt/checkpoint/seed/configuration, not arbitrary input coverage. The failed v1 RGB-assert reference was retained and was never accepted as a predecessor.

## Instrumented timing, separate from correctness

| Case | Request including audits/writes (s) | Main entry to completion (s) | API pause including recovery (s) | CPU peak (°C) |
|---|---:|---:|---:|---:|
| reference | 273.946 | 292.922 | 389.806 | 71.25 |
| repeat | 277.469 | 296.511 | 391.502 | 71.25 |
| candidate | 205.152 | 224.300 | 324.603 | 79.00 |

Candidate timing reports the slower rank; rank 1 took 203.722 seconds. Request time includes conditioning/text encoding, offload, independent prefill, all denoising/scheduler steps, VAE, output conversion, artifact writes and audits. Startup/recovery are separate. Audit work differs between cases and candidate ranks duplicate some pipeline stages. These are correctness-run measurements, not a production speedup benchmark. Candidate CPU peak was 79°C, only 1°C below the unchanged guard; production admission and thermal behavior remain separate work.

## Cleanup and bounded resources

Every window restored the exact API container `fe6ad68fd4e5ce6c83e7a747e2a6edb2c98fdb5beb4cb37badb8ccf2414e881e`, its configuration, main-checkout source mount, health and native small readiness. Redis, Postgres and frontend identities were unchanged. Final health passed and both FIFO pending/unfinished counts were zero. Candidate rank GPU contexts were reaped before restoration; no uncertain-cleanup bypass was used.

All runtime guards remained unchanged: CPU below 80°C, GPU below 90°C, one intra-op/inter-op CPU thread per rank, aggregate two-CPU quota, 128 GiB host/no swap, 22 GiB allocator per GPU, 900-second stage, 120-second collective timeout, 2,100-second total pause and 600-second recovery reserve. Retained capture, historical cases, failed v1 artifacts and successful v2 evidence total 32,698,664,784 bytes, below the 34,359,738,368-byte ceiling. No failed artifacts were deleted.

## Evidence identity

Evidence directories under `/home/andrew/Documents/Codex/2026-10-08/task-4/`:

- `trajectory-reviewed-195fc74-reference/trajectory-evidence/manifest.json`: `6798b0481a0e1766f7582abfa65372e93d800f3b76189307c66b2806fd810d2c`
- `trajectory-reviewed-195fc74-repeat/trajectory-evidence/manifest.json`: `badf9755e309b5c7741e8113f3301d6836fa298928b88b17bd2184439f4eb756`
- `trajectory-reviewed-195fc74-candidate/trajectory-evidence/manifest.json`: `6ee0e34fc8e8052b757bd9dd033471050d1ab6a8a30a5af937e141fe8c0361e2`

Each manifest binds complete artifacts and the successful comparison reports. `pause-result.json` and `restored-lifecycle.json` establish restoration. Host-local raw inspect records are not published because they can contain deployment secrets. The compact machine-readable summary is included beside this document.

Executed-head CI: [API PR 37868728251](https://github.com/ghalrym/Kadan/actions/runs/37868728251), [native PR 37868728217](https://github.com/ghalrym/Kadan/actions/runs/37868728217), [API push 37868724351](https://github.com/ghalrym/Kadan/actions/runs/37868724351), [native push 37868724251](https://github.com/ghalrym/Kadan/actions/runs/37868724251): all successful. The RGBA correction also passed 64 pinned-runtime CPU diagnostic tests before review.

Next reviewable work: persistent two-rank lifecycle and cancellation/cleanup ownership, then opt-in API/FIFO integration with real per-device resource accounting. Keep that work separate from this numerical harness and from native video/decision/audio components.
