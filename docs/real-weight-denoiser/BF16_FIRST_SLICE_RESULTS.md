# BF16 first cached-step diagnostic passed; overall protocol remains incomplete

Executed independently reviewed commit
`67b3768b9e9b1407838285d3031fc5356e2fa982` with exact-head green CI and an idle
queue. All **186 global audit boundaries** passed at unchanged `atol=rtol=.02`,
with zero violations, zero nonfinite values and maximum absolute error **0.0**.
The exact live API container was restored healthy with native model `small` ready
after **523.062 seconds (8 minutes 43 seconds)**, within the 2100-second overall
pause limit and reserved recovery envelope.

This passes only the existing first-cached-step BF16 slice. The FP32/eager and
FP32/oracle protocols remain **failed**, including original eager's later-block
oracle failures. Controlled-accumulation candidate v1 remains rejected and was
not selected. Overall BF16 protocol status is **incomplete**: steps 20 and 39,
full scheduler trajectories and decoded images have not been evaluated.

## Scope and numerical evidence

The run reused the unchanged eager capture of the red-apple prompt, seed 42,
2048×2048, 40 configured scheduler steps, first cached step with captured timestep
`0.9921875`. It used all 16,384 target rows and all 32 original trained BF16 blocks.
Capture manifest SHA-256:
`15ef4ca04d95404a0467524015097e306f8dab871eca027635d716b843b469ce`.
Every packet hash was verified before reuse; the capture was mounted read-only.
Unmodified Ulysses source SHA-256:
`8b688762e1a4900e93d1d222828909be5a287c06220ab298ec96c664007b9438`.

Passed checks:

- Independent expected global-row/local-head exchange ownership, independent
  inverse ownership, and round trip on both ranks.
- All 32 teacher-forced full-length blocks: eager replay versus captured eager,
  and Ulysses versus both eager replay and capture.
- Independent 1/4/32-block accumulated chains, checking every constituent block
  output while each path advances its own state; eager reproducibility against
  capture also passed.
- Final adaptive norm and output projection after the 32-block chain.
- Every warmup and measured cached-core timing repeat, under the same criterion.

There are 172 numerical audit boundaries before timing and 14 repeat checks.
All report zero numerical difference. This is observed equality for this captured
case, not a general BF16-equivalence claim. Internal operator values introduce no
additional acceptance gates. Per-rank and global records retain worst normalized
points, histogram counts/edges and quantile intervals under
`evidence-bf16-67b3768/`.

## Narrow cached-core timing

Both ranks agreed to timing admission from the same minimum remaining stage budget:
546.296775 seconds, above the frozen 120-second threshold. Timings use two warmups
and five measured repeats. Ulysses values below take the slower rank per repeat.

| Resident BF16 cached core | Median | Measured range |
| --- | ---: | ---: |
| Single GPU, rank 0 | 6.119164 s | 6.103740–6.125001 s |
| Ulysses, two GPUs | 3.825448 s | 3.805741–3.847260 s |

Observed ratio: **1.5996×**; cached-core latency reduction: **37.48%**.
This includes initial sharding, per-block rank consensus, communication/layout,
all 32 blocks, final norm/projection and target-output gather. Preparation,
capture/disk reads and streamed CPU audits are outside the timing window.
The single-GPU reference executes on rank 0 while rank 1 is idle but still holds
its replica. Both paths have resident blocks; this is not production component-
offload latency, complete scheduler latency, decoded-image latency, or an
end-to-end speedup claim. Runs were sequential (single then Ulysses), with only
five measured repetitions; all raw samples are retained.

NCCL logs report actual intra-node channels `SHM/direct`. No NVLink or peer-access
claim is made. The pinned image was reused without rebuilding it.

## Precision provenance

The historical capture records BF16 dtype, Torch `2.14.1+cu130`, checkpoint,
prompt/seed and timestep. **Its internal precision flags were not recorded and
remain unknown.** Current-run settings must not be imputed to that historical
capture from defaults or source intent.

For this replay, both rank identity files record matmul TF32 disabled, BF16
reduced-precision reduction enabled, cuDNN TF32 enabled, and flash/efficient/math
SDPA enabled. Profiling records the selected operator
`aten::_scaled_dot_product_flash_attention` for current eager and Ulysses paths.
These are observations about this run only. Image digest:
`sha256:eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa`.

## Resources, cleanup and exact restoration

The supervisor returned 0 after 443.987 seconds and confirmed torchrun reaped.
The diagnostic container exited 0 without host OOM, then was removed. Launcher
checks confirmed GPU-owner release and physical-memory return before restoring
only the captured API container. Its ID, image, config, host config, mounts,
command and arguments matched the pre-pause identity; unrelated container IDs
and live source hashes remained unchanged.

- API container:
  `1efcf5729f776dd90bbc0611c63a474bc696cadc0d48b9011fbb320e167b6499`.
- Restarted at `2026-10-08T23:16:27.222307589Z`; verified healthy, native `small`
  model ready. Total pause including readiness: 523.062 seconds.
- Sampled physical GPU peaks: GPU0 17,204 MiB; GPU1 16,907 MiB.
- Sampled temperatures: GPU0 81°C, GPU1 84°C, CPU 79°C, below configured guards.
- Sampled container `memory.current` peak: 2,982,989,824 bytes. This is cgroup
  accounting, not a measurement of all host memory or every shared/cache page.
- Existing capture: 22,869,681,264 bytes. New run evidence at summary time:
  683,952 bytes; combined usage stayed below 32 GiB. No new weight copies.

There were 391 resource samples; sampled peaks may miss instantaneous spikes.
Full local restoration snapshots remain outside Git because they include runtime
configuration. The committed summary records only the relevant checked identity
and readiness facts. No production source, UI, DB data or orchestration changed.

## Exact executed-commit CI and next gate

All are successful for the executed commit:

- API push: https://github.com/ghalrym/Kadan/actions/runs/37857362826
- API PR: https://github.com/ghalrym/Kadan/actions/runs/37857369563
- Native push: https://github.com/ghalrym/Kadan/actions/runs/37857362859
- Native PR: https://github.com/ghalrym/Kadan/actions/runs/37857369536

Thirty CPU harness tests passed before execution. The next protocol cases are
predeclared steps 20 and 39 using shared immutable weights and bounded streamed
contexts. Their implementation and execution still require review; this result
neither executes nor approves them. Full-image acceptance remains outstanding.
