# Opt-in API native worker (draft)

The existing API lifecycle and chat routes can select an original persistent
C++20/CUDA worker for the catalog's `small` Qwen3.5-MoE / Qwen3.6 NVFP4 checkpoint.
Python remains the default. No route, response schema, frontend, production image
or Compose service changes are required by this source draft. Source checkout is
not deployment. Follow [the controlled test plan](API-INTEGRATION-TEST-PLAN.md)
before any actual-model run or temporary service interruption.

## Configuration and packaging

Set these only on a separately approved test instance:

- `KADAN_LLM_BACKEND=native` selects this backend; omission or `python` preserves
  the existing backend. Other values fail visibly during model loading.
- `KADAN_NATIVE_WORKER` is an absolute executable path, default
  `/opt/kadan/bin/kadan-model-worker`. Install the reviewed binary and its runtime
  libraries together, read-only, with the test image's CUDA/host ABI verified.
- `KADAN_GPU=auto` (default) or one numeric GPU index selects a single GPU.
  Admission uses the exact metadata plan, plus 512 MiB device headroom.
- Optional `KADAN_GPU_BUDGET_BYTES` is a JSON object such as `{"0":21474836480}`
  (syntax example only, **not a reviewed model/test budget**). Named devices
  replace their default logical budget; other devices keep 80% of observed free
  memory. Values must be positive integer bytes no greater than currently free
  memory. The setting is read once when the shared ResourceManager initializes;
  every admission still checks physical availability. It changes the shared
  budget for all workloads in that process, not just native chat. An isolated
  test must review its exact plan, capacity and headroom before setting it.

The adapter reserves 258 MiB for the native metadata/staging/control envelope and
512 MiB for parent tokenization/IPC, plus the exact device envelope, in the API's
shared resource manager. These are accounted budgets, not OS memory limits.
The test container also needs an explicit host-memory cap. Other GPU processes
are not governed by this process-local ledger.

Compile-only example (does not execute CUDA inference):

```sh
cmake -S native -B /tmp/kadan-api-native-cuda -DCMAKE_BUILD_TYPE=Release \
  -DKADAN_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /tmp/kadan-api-native-cuda --target kadan-model-worker --parallel 2
sha256sum /tmp/kadan-api-native-cuda/kadan-model-worker
```

### Reproducible isolated test image

[native/packaging/Dockerfile](packaging/Dockerfile) is a separate multi-stage
build from repository-root source. It does not alter the default API Dockerfile
or Compose. The builder compiles only `kadan-model-worker`, SM86, Release, with
**two jobs**. The final image combines the worker, its CUDA 13 runtime library
and the reviewed API source with an existing full API dependency image. C++ and
GCC runtimes are linked statically; the worker's shell launcher scopes its CUDA
library search path without changing Torch's library environment. Build-time
`ldd` checks reject missing libraries/GLIBC versions without executing the worker.
The image records worker/runtime hashes in `/opt/kadan/SHA256SUMS`, linkage in
`/opt/kadan/LINKAGE.txt` and the source revision in its OCI label.

The helper now has concrete pinned inputs for this server:

- CUDA builder: NVIDIA `13.0.2-devel-ubuntu22.04`, **linux/amd64** manifest
  `nvidia/cuda@sha256:6b6617592b94e7dcc6ffbe6d00720eed27bc6e3b4f06b26b93b4070c31f57391`.
  This was resolved with read-only `docker manifest inspect`; image layers were
  not pulled. GCC/make/CUDA come from that immutable devel image.
- CMake **3.31.6**, official Linux x86_64 archive, SHA-256
  `5a1133ff103c71eb5120e2cc3de922733e7d8a26a98ae716397e8676adb367bf`.
  The Dockerfile downloads it and verifies the checksum before extraction.
  Checksum source: [Kitware release checksums](https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-SHA-256.txt).
  No apt/pip dependency resolution or unpublished toolchain image is needed.
- API dependency base: existing local image
  `sha256:985fc03e9ad5fee38fc50b5f088305c9ab94c9a7b3ac53addf2b20622f7c4e4e`,
  inspected as Linux amd64, tagged `kadan-api:monitor-178798c`, **no RepoDigests**.
  This is a local image ID, not a registry manifest digest. On `--build`, the
  helper verifies that exact local ID, gives it an ID-derived local tag, uses
  that tag with `--pull=false`, and rechecks its identity after the build. If
  another process changes the tag during the build, discard the output image.
  It replaces `/app/api` with the reviewed source; it does not update any
  container using the original image. The input supplies the installed API
  dependencies already used for CPU integration tests. Any future requirements
  change requires a separately rebuilt/reviewed API dependency base.

Both image inputs may be overridden explicitly; toolchain overrides must be
registry digest references, while API overrides accept an exact local image ID
or registry digest reference. The default local base must exist on the build
host; missing identity fails before compilation. This is a reproducible
source/build-input route, not a claim of bit-identical compiler output or a
successfully built/tested image.

After committing/reviewing the packaging files, from the repository root:

```sh
SOURCE_REVISION=$(git rev-parse HEAD) # inspect/approve this exact full commit
python3 -B native/packaging/build_test_image.py \
  --revision "$SOURCE_REVISION" --tag kadan-native-api:review
# Default prints commands only. To build, execute the same command
# with --build appended. It builds an image only; never runs API/CUDA/services.
```

The helper streams `git archive <exact SHA> api native` into Docker, so only
committed source enters the context; untracked deployment files, credentials,
local model weights and dirty edits are excluded. Review the committed tree as
usual. It fails if that commit lacks the packaging Dockerfile. Build failures
stop immediately; no push, GPU access, model mount, service startup or production
replacement is performed. Record the resulting image ID, both input identities,
source commit and embedded hashes in the separately approved test run manifest.
`KADAN_LLM_BACKEND=native`, exact reviewed budgets/capacity, isolation and service
stop/restoration remain **run-time decisions**, not image defaults.

CPU-only helper validation (no Docker or numerical backend imports):

```sh
python3 -B -m unittest discover -s native/packaging -p 'test_*.py'
```

Use an isolated writable model-state root with no saved selection at startup,
private Redis/output state and loopback listener. After approval, expose only
the approved checkpoint read-only within that root. The existing API auto-loads
a saved selection; never point a test instance at production writable state.
The configured context is honored exactly and never silently shrunk. The recorded
capacity-1 plan needs **20,897,997,312 bytes arena + 536,870,912
bytes headroom = 21,434,868,224 device bytes**. The recorded default logical GPU
budget, **20,014,117,683 bytes**, rejects even capacity 1. An actual chat capacity
needs a fresh metadata plan and additional free memory; the existing 65,536-token
setting is not an approved test capacity. Review exact capacity and budget
changes explicitly or reject the load; no budget/context is silently changed.

### Preview the separate approved run

`native/packaging/preview_test_run.py` **only prints JSON argv**; it has no
subprocess executor and cannot start a container. After the image is built,
supply its exact local image ID and the separately reviewed isolation settings:

```sh
python3 -B native/packaging/preview_test_run.py \
  --image "$BUILT_IMAGE_ID" --state-dir "$EMPTY_PRIVATE_MODEL_STATE" \
  --checkpoint-dir "$APPROVED_READONLY_SNAPSHOT" \
  --checkpoint-relative "$CATALOG_SNAPSHOT_RELATIVE_PATH" \
  --env-file "$PRIVATE_TEST_ENV_FILE" --network "$PRIVATE_TEST_NETWORK" \
  --gpu "$APPROVED_HOST_GPU" --host-bytes "$APPROVED_HOST_LIMIT" \
  --device-bytes "$APPROVED_DEVICE_BUDGET" --port "$PRIVATE_LOOPBACK_PORT"
```

These inputs deliberately have no operational defaults. The state directory
must be new/private with **no saved selection**, separate from the approved
checkpoint source. The relative checkpoint destination must match the catalog
layout; the generator does not discover or read weights. The reviewed network
must be a dedicated internal `kadan-native-review-*` network with private Redis
and Postgres; the env file supplies only those test connection settings. Do not
use production database/Redis credentials or existing production writable
volumes. Prepare empty test databases through approved schema migrations only,
never mock records/seeds. The generator validates syntax and bounds, not the
truth of these isolation facts: verify them against the controlled test plan.

The preview exposes only the selected host GPU (mapped to device 0 inside),
binds HTTP to loopback, caps CPU at 2, sets explicit memory/swap/pid limits,
drops capabilities, makes the container root read-only, mounts only private
model state writable and the selected checkpoint read-only, and disables hub
network downloads. It neither creates the network/dependencies nor stops or
restores production. No run command may be executed until the exact stop,
isolation, actual-model test and restoration plan has separate approval.

## Lifecycle and protocol

1. The Python parent launches `--plan ROOT CAPACITY METADATA_BYTES`. This reads
   configuration and checkpoint headers, not payload tensors or CUDA devices.
   It returns `plan 1 HOST_BYTES ARENA_BYTES VOCAB CAPACITY MIN_STAGING_BYTES`.
2. The parent admits and leases paired RAM/VRAM reservations, loads the local
   tokenizer, then starts `--serve ROOT DEVICE CAPACITY HOST_BYTES DEVICE_BYTES
   HEADROOM_BYTES`. Only this second mode loads payloads/initializes CUDA.
3. `ready 1 VOCAB CAPACITY ARENA_BYTES HOST_BYTES` must match the admitted plan.
   A request sends `reset`, each prompt ID with `step ID 0`, then generated IDs
   with `step ID 1`. Replies report selected token, EOS and committed progress.
   Early prompt predictions of EOS do not end prefill. The final prompt output
   is the first generated token. A P-token prompt and N non-EOS output tokens
   need P+N-1 steps when ending at the output limit.
4. Close/EOF releases the native model and verifies its reservations are zero
   before emitting `closed 0`. Errors are terminal. The parent drains bounded
   stderr, validates bounded replies, supervises timeouts/cancellation, and
   reaps the owned process before releasing global reservations. Child ownership
   is published before IPC setup; reload and generation share the same gate. Uncertain exit
   retains both accounting and lifecycle ownership for an explicit unload retry.
5. Eviction closes both native reservations together. A later generation reloads
   through the same admission path; an explicitly closed adapter never reloads.

The worker does not fork children or use external inference engines. The protocol
carries token IDs only; Python owns the existing local chat template, text
streaming, HTTP lifecycle and resource coordination. No Strata/FreeToken code or
kernels are used. Other modalities retain their existing adapters.

## Limits and validation

This is a greedy, single-GPU, all-expert-resident path using the existing
correctness kernels. Prefill is sequential token stepping. Prefix reuse is
explicitly reported unavailable; every request resets model state. The current
API output cap remains 256. Dual-GPU scheduling, optimized prefill, sampling and
50+ decode tokens/sec are not implemented or demonstrated by this integration.
No numerical kernel changes are part of this draft.

CPU tests use generated metadata and fake engines/subprocesses for protocol,
EOS/progress, failure, stderr backpressure, timeouts, cancellation, admission,
eviction/reload and cleanup retry. CUDA compile/link is not GPU correctness or
performance evidence. The prior pinned-HF synthetic tie case remains a recorded
failure: canonical native tie ordering differs from that HF observation. Neither
the passing synthetic cases nor this API wiring establish actual-model parity.
