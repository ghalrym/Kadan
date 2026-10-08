# Staging reviewed MR119 on the home API

MR119 was activated for review with explicit user permission on2026-10-08. See
[HOME-API-RESULT.md](HOME-API-RESULT.md) for the exact running revision, successful
HTTP verification and rollback. Keep MR119 unmerged. The instructions below
describe staging constraints; preserve local files and use the recorded backup
when changing the authorized API-only override.

## Explicit component selection

Set `KADAN_IMAGE_OFFLOAD=component` to select the measured whole-component path.
The default is `sequential`. Both preflight and provider construction read the
same setting; invalid values fail before handoff. Changing mode closes the old
idle pipeline before constructing a new one. Compose forwards the setting from
`.env` or the invoking environment. This does not change precision, image size,
step count, guidance, or VAE tiling. Compilation experiments remain harness-only.

## Minimal dependency and hot-reload steps

1. Keep a separate checkout of the reviewed MR head and its generated API client.
   The measured implementation is101bc45; later evidence/config commits are
   independently reviewable. Mount the selected checkout's `api/` at `/app/api`
   read-only using the existing Uvicorn reload workflow. Existing bind mount
   source changes need a one-time API container recreation; later Python edits
   then hot-reload normally. Do not copy files into the live checkout blindly.
2. The running image `kadan-native-api:review-ba380d9` has Diffusers0.37 without
   QwenImage21Pipeline. The cached `kadan-api:image52-7392636` immutable image
   `eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa` already contains
   the pinned Diffusers8d3c30b and compatible Torch/vision/Transformers dependencies.
   Use that verified cached runtime for an isolated test service, or build only
   the missing pinned Diffusers dependency layer over the live runtime for a
   deliberate migration. A full routine API/base-image rebuild is unnecessary.
   Never rely on Python hot reload to replace an installed dependency.
3. Mount the already-built resident worker and matching libcudart13 from
   `acceptance-119-cache/native/` into a separate native directory. Worker SHA256
   `02309c0817596588b52f86586748fe4120ce81c3b652607fe6a27016e2496e58` matches unchanged
   native sources since2fab5f1. Its wrapper scopes LD_LIBRARY_PATH to the child.
   Only rebuild `kadan-model-worker` when native sources change. Do not replace
   the live legacy binary in place.
4. Configure `KADAN_LLM_BACKEND=native-resident`, the mounted
   `KADAN_NATIVE_WORKER` wrapper, and `KADAN_IMAGE_OFFLOAD=component` explicitly.
   For the proven single-card setup, expose physical GPU1 only, mapped to logical0,
   and set `KADAN_GPU=0`, `KADAN_IMAGE_DEVICE=cuda:0`. Both image and text then use
   the same physical card through the reviewed FIFO/residency handoff.
   The current live override uses text GPU0/image GPU1, so copying it unchanged
   does not reproduce the measured same-card switching arrangement.
5. For a sidecar test, use a separate Compose project/service name, dedicated
   Redis with a distinct queue/lock/output directory, and a different localhost
   API port (e.g.8001). Mount existing checkpoints read-only; no download or
   database fixture insertion is needed. Keep the live8000 API and its Redis
   consumer untouched. The acceptance harness injects96GiB host/22GiB GPU budgets
   and a96GiB/no-swap cgroup; production API normally derives host capacity from
   80% available memory, so don't claim it has that exact96GiB parent budget.
6. Verify imports, worker hashes/linkage and `--plan-resident` with no GPU first,
   then `/health`, image catalog and the existing file/queue routes. For a GPU1
   sidecar, require a fresh headroom check and preserve acceptance guardrails.
   Do not start another model service while the bounded benchmark owns GPU1.

Switching the existing home API's image/environment/native bind requires an
API-only recreate/restart, not merely Uvicorn reload. That interrupts its GPU0
worker; it was performed only after explicit user approval; see the result above. A staged sidecar enables testing
without that interruption. Keep Redis/Postgres volumes and current model data
unchanged. The original image/backend/worker bind and source checkout remain the
rollback inputs. The environment setting is selectable now; promotion or default
changes should follow review of the measured limits and intended device layout.
