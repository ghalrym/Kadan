# Read-only live switching readiness — 2026-10-08

Snapshot at 18:14:56 UTC. No model execution, download, deployment, live restart,
weight modification, hardware setting change or image rebuild was performed.

## Existing checkpoint and earlier execution evidence

`kadan_model_data` already contains
`qwen-image-2.1-d26bb61231c349cf6b7896fa83353113880e1ba3`.
The normal `ModelManager._checkpoint_complete()` validator returned **True** using
the MR119 catalog entry. All 26 recorded files exist with their recorded sizes;
required component assets and symlink checks pass. Total recorded bytes:
33,131,615,131; seven safetensors shards total 33,115,613,408 bytes (30.8413 GiB).
This is the normal metadata/completion check, not a new full-payload hash scan.
**No new image checkpoint download is required.**

PR52's body records synthetic/browser/import evidence only, but later local
records show a real attempt on 2026-10-07. `image52-work/cancel-result.json` and
`generation-cancel.log` record 40/40 denoising steps in 18m30s, user cancellation,
and no published PNG. Recorded GPU1 peak usage was 3,596 MiB, maximum temperature
79 C, and minimum available RAM 389.32 GiB, with no monitor guard event. These
are historical measurements for that cancelled attempt, not successful image
quality validation or proof of the new switching path.

Evidence directory:
`/home/andrew/Documents/Codex/2026-10-06/task-4/image52-work/`.
The later browser handoff record says the prior idle thermal-counter discrepancy
was not grounds for an indefinite blanket hold; use fresh telemetry for a future
bounded test. No safety/thermal settings were changed during this readiness check.

## Runtime inventory: reuse existing images

Live API: `kadan-native-api:review-ba380d9` (`32f6aeb88e96`), healthy.
Installed Torch 2.14.1, torchvision 0.29.1, Transformers 5.17.0, Pillow 11.3.0 and
Accelerate 1.10.1 match this bridge's relevant pins. Its Diffusers 0.37.0 lacks
`QwenImage21Pipeline`; source hot reload alone cannot supply that dependency.

However, cached `kadan-api:image52-7392636` and `kadan-api:image52-candidate`
both identify image `eed6c0b12088`, already containing Diffusers 0.41.0.dev0 at
exact commit `8d3c30bfda9b511c00992f40cff4170a5502814d` and the same other versions.
Its pipeline SHA256 is
`6985b2f1f25e8dd09ef85c867ebcad184b981f10c0b4f80757ca9cc135b8fa5e`.
It also already has CUDA runtime 13 at
`/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`.
This metadata inspection used an ephemeral container without network, GPU or
model mounts. **A full API dependency rebuild is unnecessary for an isolated
probe using that cached runtime.** Alternatively, a deliberate later migration of
the live native image needs only the missing pinned Diffusers dependency layer,
followed by import and dependency checks.

The live native executable does not advertise `--serve-resident`. Build only the
MR118 `kadan-model-worker` target and required native dependencies into a separate
artifact directory, with matching runtime libraries; do not replace the live
binary. The current live executable links `libcudart.so.13`. Verify the new
binary's linkage against the cached runtime and run `--plan-resident` before a
future real request. The metadata-only legacy planner currently reports:

```text
plan 1 270532608 22282094592 248320 65536 4100
```

Thus the current 65,536-token small checkpoint arena is 22,282,094,592 bytes.
Resident mode changes storage ownership but preserves that total arena layout.

## Memory snapshot and proposed isolated test budgets

Host RAM: 473,024,593,920 bytes total; **424,539,766,784 available** (395.383 GiB).
Swap is unused and is excluded from the proposed budget. The live API cgroup has
no finite memory.max limit. GPU0 remains occupied by the unchanged live native
text worker; GPU1 is the candidate test device.

| Physical device | Used | Free |
| --- | --- | --- |
| GPU0, `GPU-e30b6419-2c6d-f550-61d6-16166a920dac` | 21,779 MiB | 2,346 MiB |
| GPU1, `GPU-2a2378dd-08c1-6f69-6317-a253d90e76b3` | 862 MiB | 23,254 MiB |

Propose a separate bounded probe exposing **only GPU1**, mapped to logical GPU0,
with `ResourceManager(103079215104, {0: 23622320128}, probe=probe_memory)`:
**96 GiB host and 22 GiB GPU**. These are explicit harness-injected capacities,
not a nonexistent host-budget environment option. The ordinary API computes its
host budget from 80% of available RAM when initialized. Never transplant the live
GPU0 capacity override into this probe. Use one shared parent manager, separate
Redis namespace and writable queue-lock/output directory, and mount existing
checkpoints read-only. The live Redis consumer's lock must not be reused.

| Envelope | Bytes | GiB |
| --- | ---: | ---: |
| Native host/control/cache/Python | 1,077,936,128 | 1.0039 |
| Native model arena, capacity 65,536 | 22,282,094,592 | 20.7518 |
| Native context retained at park | 536,870,912 | 0.5 |
| Image host, twice weights + workspace | 74,821,161,408 | 69.6826 |
| Image output or edit-validation scratch | 1,073,741,824 | 1 |
| Image sequential GPU workspace | 8,589,934,592 | 8 |
| Python image context, process lifetime | 536,870,912 | 0.5 |
| Text-phase peak GPU after image context exists | 23,355,836,416 | 21.7518 |
| Image-phase peak host with native backing and output | 76,972,839,360 | 71.6865 |

The complete image weights plus workspace cannot fit one 24 GiB card, so this
plan selects sequential CPU offload. During image execution both context envelopes
stay charged, for 9 GiB total GPU admission. During the final text request the
image model must be parked; its Python context remains charged. Fresh physical
probes remain mandatory at admission; these figures do not authorize overruns.
The 8 GiB image workspace is still an admission estimate, not a proven upper
bound for every image phase or resolution.

## Remaining prerequisites

Finish MR119 review/CI after the publication and preflight corrections. Prepare
the separate reviewed resident binary and an isolated real-leaf queue harness on
the cached PR52 runtime, with the exact capacities above and fresh telemetry.
Use one square image, count 1, fixed seed 42 between short text requests; allow a
bounded window consistent with the historical 18m30s attempt and verify an actual
published PNG plus final text response. No successful real image output or
text/image/text hardware result is claimed yet. Preserve the live server and
its data throughout; no missing checkpoint or mandatory full-image build blocks
preparing that isolated test.
