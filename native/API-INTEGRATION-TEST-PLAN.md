# API integration: controlled test and restoration proposal

**Source-only proposal; not execution authorization.** The API draft stays
unmerged and checked out for Andrew's review. Merge of synthetic tooling #113
(`49d32b0b4fffa65baceda2bd7179137893fd3d4e`) does not authorize installed-model
payload reads, inference, benchmarking, API interruption or deployment. Those
operations require approval of the completed run manifest below. No commands in
this document have been executed while preparing it.

The direct path is an opt-in native adapter using the existing API contracts,
with CPU/synthetic contract tests first, then a separately approved **temporary
test instance while the original API is stopped**. This avoids building a broad
maintenance framework merely to run one reviewed experiment. It does not solve
concurrent cross-process resource ownership for a later production rollout.

## Source findings and packaging

- `api/server.py::lifespan` starts `memory_manager` and then
  `RuntimeManager.start()`. The latter reads saved selection and starts loading
  in the background. Starting an API with the production model root is therefore
  potentially actual inference allocation before any test request.
- `api/memory_manager/queue.py` binds the Redis namespace to an inference-lock
  identity. A temporary API needs **its own Redis instance and writable state
  root**; do not share production Redis, replace its lock or cancel/delete jobs.
  `selection.json` and `context.json` belong to the model root. Isolate those
  files and all output directories, not just the HTTP port.
- `compose.yaml` and `api/Dockerfile` build from `./api`. Root `native/` sources
  and binaries are outside that context. Existing images cannot acquire the
  C++ worker from a source checkout or an opt-in environment setting alone.
- The local override names image `kadan-api:monitor-178798c` and revision
  `178798c277ac3985b5ca836191b3430a4d71d55f`. This is **reported configuration,
  not a fresh verification of the running deployment**. Other local deployment
  notes contain historical revisions; none is a safe identity for a stop command.
- The frontend proxies to the production API. Keep it pointed there throughout;
  while that API is stopped, submissions fail rather than reaching the test API.
  Existing streams disconnect. Do not change proxy/firewall rules or expose the
  test listener on the LAN. An operator must stop submissions before draining;
  a quiet interval alone is not an atomic gate.

Build from the repository root with at most two compiler jobs. CPU checks need
no model volume, API lifespan or CUDA initialization:

```sh
cmake -S native -B /tmp/kadan-api-native-cpu -DCMAKE_BUILD_TYPE=Debug -DKADAN_ENABLE_CUDA=OFF
cmake --build /tmp/kadan-api-native-cpu --parallel 2
ctest --test-dir /tmp/kadan-api-native-cpu --output-on-failure
python3 -B -m unittest native.tests.reference_capture.test_artifacts
# In the repository's reviewed CPU test environment, with isolated test Redis:
python -m unittest discover -s api/tests -v
```

Use the CPU dependencies/test-Redis setup in `.github/workflows/api-tests.yml`;
never direct tests at production Redis or Postgres. Add adapter tests for default
backend preservation, explicit opt-in, unsupported settings, spawn/handshake,
framing bounds, EOS, cancellation, timeout evidence, failed cleanup and held
resource ownership. Use fake child processes/generated fixtures. A passing CPU
suite does not measure CUDA performance. If public schema changes, regenerate
OpenAPI and the client as required by AGENTS.md; otherwise verify no schema drift.

For the eventual isolated GPU test, choose one reviewed packaging route:

1. Build the reviewed native target separately using the pinned CUDA toolchain,
   then mount its immutable build/install directory read-only into a disposable
   API test container. Pin executable hash, dependent shared libraries and image
   ID; verify the executable path and dependencies without executing a GPU test.
2. A separate root-context test Dockerfile can compile native code and copy its
   outputs into a pinned API runtime image. Review CUDA/compiler/runtime ABI and
   library search paths. Build with two jobs; use a unique test image tag and pin
   the image ID. Do not edit the production Compose override or replace its tag.

The final draft must document its actual opt-in variable, executable option and
supported target rather than treating either packaging option as implemented.
Compile-only CUDA checks may be reviewed separately; no parity/benchmark binary
is an ordinary image build step. Source checkout is not deployment.

## Manifest required before approving a window

Fill and review every item; unresolved fields block execution. Never guess names
from old notes. Retain a private, mode-0700 evidence directory; redact secrets
from any shared configuration report.

| Item | Required pinned value |
| --- | --- |
| Source | Exact unmerged API head, native source revision, CPU/CI results and independent review |
| Original API | Container ID, image ID/digest, command, Compose project/file identities, restart policy, service epoch, GPU assignment, environment/config and mounts |
| Original state | Saved model/revision, configured/effective context, output limit, lifecycle, active jobs/requests/downloads, memory reservations; reported prior context 65,536 is not a current assertion |
| Isolation | Exact API stop/start command targeting that container only; confirmed child/process ownership; temporary container names/IDs, Redis, state/output roots and unused loopback port |
| Artifact provenance | Worker/reference executables, libraries/image hashes, metadata and approved payload-hash procedure; snapshot writer exclusion |
| Exact test | Input IDs or rendered prompt/tokenization hash, context capacity, batch, decode count/EOS handling, seed/sampling policy, GPUs, numerical criterion and allowed repetition count |
| Budgets | Native plan for that exact capacity, host/device reservations, physical/cgroup limits, allowed existing residents; no unrelated eviction |
| Limits | Drain, graceful stop, test, terminate/cleanup and restoration observation deadlines; approved power/temperature/telemetry thresholds and operator availability after the breaker incident |
| Restoration | Same original container/image/config/settings, expected startup auto-load, read-only readiness checks and escalation contact |

Approval must explicitly cover temporary API unavailability, payload hashing,
reference/native actual inference, any benchmark, and original startup model
reload. Keep deployment approval separate. Do not infer permission to stop
frontend, Redis, Postgres, other GPU services or desktop processes.

## Reversible sequence after approval only

1. **Prepare before interruption.** CPU tests, package checks and review must
   pass. Snapshot the manifest and original lifecycle/settings without changing
   them. Verify no downloads/snapshot writers and no pending/active work. Agree
   that clients will cease submissions. If the queue cannot drain within the
   reviewed deadline, abort before stopping; do not delete jobs to obtain zero.
2. **Stop only the verified original API.** Proposed command shape is
   `docker stop --time <approved-seconds> <verified-original-api-container-id>`.
   Resolve and review concrete arguments beforehand. Docker's stop deadline can
   escalate to KILL: include that behavior in approval or use a separately
   reviewed graceful-only procedure. Do not run `compose down`, `up`, rebuild,
   change restart policy, restart dependencies or install a firewall. A successful
   client return is insufficient: inspect container state, owned child/process
   tree and GPU residency. Require all API loaders/workers/GPU children exited,
   no API listener and no automatic restart before any test GPU allocation.
3. **Verify exclusive test ownership.** Observe unchanged unrelated residents and
   sufficient free RAM/VRAM. Stopping the API removes its producers, not every
   producer on the host. If another allocator/writer remains uncontrolled or
   appears, abort; do not evict or kill it. No production shared ledger is active
   to lend a lease: the test supervisor owns one explicit native envelope for the
   entire child lifetime and retains it through uncertain cleanup. A file lock
   that unrelated producers ignore is not sufficient proof of exclusivity.
4. **Start only the reviewed temporary instance.** Use the pinned test image,
   loopback-only listener, fresh Redis and state/output roots. Mount the selected
   checkpoint read-only only after payload permission, exposing no other models.
   Begin with absent selection so `RuntimeManager.start()` cannot auto-load;
   verify startup behavior with the draft's CPU tests. Explicitly select/load
   only the approved case in isolated state. Do not mount production selection,
   queue or output state writable. Do not run migrations or seed Postgres; if a
   route requires persistent database state, stop and review isolation first.
   Restrict submissions to the operator/test client. Exposing all API routes
   without controlling callers would reintroduce other-modality admission risk.
5. **Correctness before throughput.** Run the individually reviewed case once,
   preserve exit status/stdout/stderr even on timeout, and require complete
   artifacts plus cleanup evidence. Exact comparison remains 0/0; no tolerance
   widening or changed expected output. #113's synthetic reference wrapper
   rejects actual dimensions; it is not an actual-model reference command.
   Actual support and supervision need separate review. Preserve the known raw
   HF router-tie failure; canonical native routing and pinned-HF compatibility
   are separate claims. Reference-only layer hashes cannot locate native errors.
6. **Benchmark only under its own exact approved case.** Do not turn a successful
   single BOS forward into permission for autoregressive decode or a sweep.
   Specify maximum generated tokens, prompt length, context, single concurrency,
   maximum runs and hard duration before execution. Report load time, prefill,
   time to first token and steady decode separately. Early EOS/too few tokens
   makes the decode measurement insufficient. No background benchmark loop,
   dual-GPU expansion, power-limit change or automatic retry.
7. **Stop and verify temporary cleanup.** Close its listener; terminate the
   recorded test API and every supervisor-owned native child, then verify exit,
   no new compute processes and GPU allocation returned to the saved stopped-API
   baseline within telemetry precision. Docker client timeout is not container
   cleanup. Keep reservations/failure fencing while any child is uncertain.
   Never overlap original auto-load with uncertain native allocations. Do not
   reset GPUs or kill unrelated processes to force a clean result.
8. **Restore the original instance once cleanup is known.** Proposed command
   shape: `docker start <same-original-api-container-id>`. Do not use `compose up`
   from the draft checkout, recreate the container or load the draft image.
   Startup may restore the saved model and read its payload; approval covers
   that reload. Verify original image/config/settings/context, read-only health
   through the unchanged frontend, queue status, lifecycle readiness and memory
   ownership. Do not generate a token to prove restoration. If starting the same
   container cannot restore it, stop and report; a new deployment/restart loop
   is not an authorized fallback. Leave the source branch checked out and PR
   unmerged for morning review, independently of the restored running image.

Numerical failure still calls for restoration after verified cleanup. Unknown
cleanup requires the original API to remain stopped until ownership is resolved
and the operator is informed. No deadline guarantees an upper bound on outage;
termination, GPU cleanup and reload may remain uncertain beyond it.

## Budget and performance gates

The prior capacity-1 plan records native arena **20,897,997,312 B**, device
headroom **536,870,912 B**, device envelope **21,434,868,224 B**, and native host
envelope **271,525,888 B**. These are single-BOS values, not a chat/decode budget.
The observed production logical GPU budget **20,014,117,683 B** is smaller than
that envelope. A stopped-service standalone ledger does not justify raising the
API's budget or disabling admission. The draft must either reject insufficient
admission or expose a separately reviewed explicit test budget within measured
physical/cgroup limits. Recompute the exact planner result for any prompt/decode
capacity before approval; include tokenizer/API/supervisor memory outside the
native ledger. The held reference proposal has a separate 40 GiB CPU cap.

The **50+ decode tokens/sec on dual RTX3090s** objective is unmeasured. The current
all-resident single-GPU path does not demonstrate dual-GPU execution or that
performance. Full-model matrix-vector bandwidth, many kernel launches, recurrent
state/attention cost, host/device synchronization and process/API transport remain
risks. A per-request process/load design would add full load latency; amortization
requires a persistent reviewed worker, not a throughput assertion from synthetic
correctness. Support for other modalities remains unchanged.

The bounded alternative while approval is pending is the checked-out API draft,
CPU fake-worker/synthetic tests, CI, source review and packaging instructions only.
A later production rollout needs its own deployment review, resource-admission
policy and rollback; this temporary test plan does not authorize it.
