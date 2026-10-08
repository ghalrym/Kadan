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

For the reviewed test packaging path, mount the compiled executable and required
libraries read-only into a disposable API container built from the reviewed API
source. The default API Dockerfile's `./api` context excludes `native/`; neither
an environment variable nor checking out this branch installs the binary. Pin
that container's image, worker hash and shared-library ABI in the run manifest.
No new production packaging or service startup is implemented here.

Use an isolated writable model-state root with no saved selection at startup,
private Redis/output state and loopback listener. After approval, expose only
the approved checkpoint read-only within that root. The existing API auto-loads
a saved selection; never point a test instance at production writable state.
The configured context is honored exactly and never silently shrunk. The
existing 65,536-token setting and 80% budget may not fit this resident single-GPU
path. Review a new explicit test capacity and its metadata plan, or reject load.

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
   reaps the owned process before releasing global reservations. Uncertain exit
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
