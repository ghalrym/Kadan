> Historical benchmark record: temperature readings and thresholds below describe the original machine-local procedure, not Kadan runtime policy. Current benchmark temperature monitoring belongs outside the repository; see [external monitoring](/docs/EXTERNAL-BENCHMARK-MONITORING.md).

# Reviewed v1 reference: output-contract failure

Executed source: `0a673c887897bd4e3f3efb6ba79321157eb922a2`, from the sole main checkout. API/native exact-head CI passed before execution.

The first admission stopped at CPU 61.25 C against the unchanged below-60 C admission gate. The API remained running. A fresh attempt after cooldown began at 2026-10-09T01:04:12Z.

The reference completed all 40 denoising/scheduler steps, retaining 40 predictions, 40 post-step latents and the initial latent. The raw VAE finite check passed. The run then failed the asserted float32 NHWC RGB shape in image postprocessing. Pinned VAE configuration has four output channels and the processor preserves RGBA; the harness incorrectly required three. This is a failed reference, not a passed full-image result. No output PNG or successful manifest exists; repeat and candidate were not run.

CPU peak was 70.25 C. The supervisor reaped its process and reported 307.856 seconds to failure; this is failed-run instrumented wall time, not generation latency or a speedup measurement. Total API pause including restoration was 387.626 seconds, within 2,100 seconds. The exact container `fe6ad68fd4e5ce6c83e7a747e2a6edb2c98fdb5beb4cb37badb8ccf2414e881e` was restored with identical configuration/mounts, health and native small readiness. Supporting services were unchanged.

Evidence remains outside source under `/home/andrew/Documents/Codex/2026-10-08/task-4/trajectory-reviewed-0a673c8-reference` (admission) and `trajectory-reviewed-0a673c8-reference-r2` (failed execution). No captured tensors were deleted. Together they add 170,352,768 bytes to the next admission budget.

The follow-up protocol compares native RGBA, retaining RGB mean bounds and independently checking alpha. It increases only the artifact allocations needed for four channels and a bounded RGBA PNG, remaining below the existing 32 GiB aggregate storage ceiling. It requires fresh source review before GPU execution.
