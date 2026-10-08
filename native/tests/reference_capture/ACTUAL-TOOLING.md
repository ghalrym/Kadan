# Actual-reference tooling draft — no execution approval

This is test-only support, based on master independently of API draft #114.
`capture.py` and its synthetic dimension/shard/timeout guards are unchanged.
The backend now accepts an explicit frozen `Limits` object; default synthetic
limits remain 128 MiB host / 4 MiB request. Actual limits are fixed at 40 GiB /
512 MiB. No equations, router selection, expected output or tolerance changes.

## Entrypoint and required pins

`python -B -m native.tests.reference_capture.capture_actual` requires the explicit
`--execute-approved-actual-reference` flag plus `--snapshot-root`, `--manifest`,
`--manifest-sha256` and a nonexistent `--output-dir`. There is no supplied actual
manifest or shard digest. Do not create one until payload access is authorized.
The SHA256 argument pins the entire approved JSON, including these fields:

- `schema: 1`, `execution_authorized: true`, `snapshot_writer_exclusion: true`.
  These record externally established approval/exclusion, not a locking mechanism.
- `snapshot: small-1355db6a052410cfd62085d94b58866fd0f2c3c5`,
  `revision: 1355db6a052410cfd62085d94b58866fd0f2c3c5`.
- `input_ids: [248044]`, `capacity: 1`, `timeout_seconds: 1800`,
  `host_bytes: 42949672960`, `request_bytes: 536870912`.
- `source_revision`: exact reviewed Git commit; `wrapper_sources`: exact mapping
  of every sibling Python filename to SHA256; `backend_pins_sha256`: SHA256 of
  backend-pins.json; `image_id`: exactly the image recorded in that file.
- `files`: mapping of every admitted snapshot filename to `{size, sha256}`.
  Requires reviewed config/generation/tokenizer-config digests, the index,
  tokenizer.json, every index-listed numbered model shard, and optionally
  special_tokens_map.json. No extra snapshot entries are admitted. Metadata is
  bounded to 1 MiB, index to 16 MiB, inventory to 128 files and total to 32 GiB.
  Future metadata-only review must reconcile the installed inventory; do not
  silently remove files or broaden admission if it differs.

Source/backend validation precedes shard hashing or Torch import. Shards stream
through <=1 MiB buffers with no symlinks, exact lengths, SHA256 and descriptor/path
identity checks. All files are rehashed after the single forward; manifest/source
pins are checked again. External writer exclusion remains required because checks
cannot prevent a writer changing and restoring bytes during inference.

The process requires cgroup v2 memory/swap/CPU/PID ceilings, no GPU devices or
visibility and only loopback networking. It sets offline mode and two intra-op /
one inter-op threads. Existing BF16, backend provenance, cache/call counters,
router/layer diagnostics and zero-reservation checks remain in force. Artifacts
are exclusive 0600 files in a new 0700 directory; capture size is 993,316 bytes,
diagnostics <=8 MiB. Failure/timeout invalidates every capture even with DONE.
The 1,800-second alarm includes hash/load/forward/post-check/publication work.

Raw HF tied routing remains distinct from canonical native routing. Native
lower-ID ordering is unchanged; observed HF [2,3] versus canonical [0,1] remains
an exact-compatibility failure, never an expected output substitution. Actual
selected token and logits are unknown. Comparison still requires exact 0/0.

## Outer supervisor and ownership

`supervisor.supervise` is the bounded state machine; `container_stage` supplies
an intentionally narrow CPU-reference Docker transport. Its CLI requires
`--execute-approved-reference`, immutable `--container-id`, `--run-id`, manifest
and digest, and absolute `--source`, `--snapshot`, `--evidence` paths.

It only accepts an already-created `/bin/sleep infinity` container with the
manifest's pinned image, `kadan.reference.run` label equal to the run ID, runc,
network none, no GPU devices, read-only root, dropped capabilities, no-new-privileges,
no restart, <=2 CPUs, <=40 GiB RAM/no extra swap and <=256 PIDs. The only bind mounts
are source -> /work read-only, snapshot -> /snapshot/<exact-name> read-only and
private 0700 evidence -> /evidence writable. Manifest must reside in evidence.
CPU source does not create containers/networks or manipulate production services.

The supervisor verifies ownership before launching exactly one explicit exec.
Raw child streams use exclusive files with 8 MiB per-file ceilings. It validates
exit, full artifact/hash/diagnostic completion, Docker OOM and cgroup memory.events.
On deadline/failure it requests TERM, waits up to five seconds, then requests KILL
if needed. Every RPC is bounded. It then requires five consecutive one-second
observations of actual exit, reaped exec client, no owned processes/compute,
released memory and restored baseline. Deadline/kill RPC success never proves exit.
SIGTERM/interrupt initiates cleanup; SIGKILL or host failure still requires an
external operator to retain ownership.

`restoration_permitted` means **only the cleanup prerequisite**, never authorization
or an action. No restoration API is implemented. Numerical failure with proven
cleanup can allow a separately authorized driver to restore; uncertain cleanup
always returns false. The CLI exits 2 on uncertainty and 1 on a clean failed run.

## Remaining technical and operational gates

- Independent review of exact source/backend/image pins and all new tooling.
- Actual payload hash manifest and writer exclusion, separate execution approval,
  electrical/operator attendance, fresh physical/cgroup headroom.
- The concrete CPU transport conservatively requires a final empty cgroup and
  memory.events sample. Docker engines that remove the cgroup immediately cannot
  satisfy this gate: **hold; do not restore**. A reviewed retained-cgroup observer
  or equivalent kernel-backed lifecycle evidence is required for those engines.
  This behavior was tested with fake subprocesses, not a real model/container window.
- GPU native/benchmark/API stage transports and baseline telemetry integration
  remain outside this CPU transport. Do not use its zero-compute CPU evidence for
  GPU work. The generic supervisor supports such observers but none is supplied.
- Test-only correctness/comparator overlay image is not built or repinned here;
  the existing native/API images are preserved. Real prompt IDs remain unpinned.
- Service coordination and restoration authorization remain pending; no deployment,
  service stop/start, actual checkpoint read/hash, GPU run or test network occurred.

## CPU verification

Run the stdlib suite (also registered with native CTest):

```
python3 -B -m unittest native.tests.reference_capture.test_artifacts \
  native.tests.reference_capture.test_actual \
  native.tests.reference_capture.test_supervisor \
  native.tests.reference_capture.test_container_stage
```

Only invented manifest metadata, generated byte fixtures and fake subprocesses
exercise actual-entrypoint gates and supervisor failures. Existing numerical
fixture runner remains unchanged and is run separately in the pinned CPU-only
image without model mounts. A passing gate suite does not establish actual-model
correctness or validate a real container cleanup observation path.
