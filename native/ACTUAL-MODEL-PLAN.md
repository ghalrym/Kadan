# Actual-checkpoint validation: review proposal, not execution authorization

Prepared after the four synthetic GPU stages at
`a9f165edac164ac2f49a00cc1209d1745ae983b8`. Andrew's approval to merge #112 and to
open an unload/test/reload window is **pending**. No actual weight payloads,
reference inference, actual-model GPU execution, unload/load, network gate or
service mutation was performed while preparing this document.

This is a one-BOS correctness experiment, not throughput validation. The native
engine remains original C++20/CUDA; the existing Python/PyTorch/Transformers
weight-only adapter is an independent test reference, not a new native dependency
or a Strata/FreeToken implementation. No production source/configuration changes,
service restart, deployment, model setting changes or other modalities are needed.

## Verified synthetic evidence

Each stage ran once with capacity 8, IDs 2/7/11, a 30-second internal watchdog,
GNU timeout 45 seconds with five-second kill grace, and exact comparator 0/0.
All processes and comparators exited zero, produced 260-byte captures with DONE,
reported zero final reservations, and had unchanged source/binary/fixture hashes.
Every capture was byte-identical to its independent reference. EOS flags were
zero, committed counts 1/2/3. GPU0 free memory was 14,409 MiB before and after
each stage; compute-process residency and API lifecycle/settings were unchanged.

| Fixture | Arena B | Device envelope B | Static forward launches | Selected IDs | Capture/reference SHA256 |
|---|---:|---:|---:|---|---|
| tiny-4 | 54,784 | 536,925,696 | 345 | 11,11,5 | `f885a374976721b598c67ec754ea40c48edc13e58c6d66443d167ce6f187809d` |
| tiny-8 | 107,008 | 536,977,920 | 678 | 2,11,13 | `c9971b61969bef725596c6ff726bea2dd014bc8a93d7c101da632aa72868a90a` |
| representative-4 | 565,248 | 537,436,160 | 705 | 9,9,5 | `133d4b640d7791304d9e90a03a50f5e2dfc0a12a893b28d0f203dbe420f1989f` |
| tiny-40 | 524,800 | 537,395,712 | 3,342 | 2,13,7 | `5ddd3dfc54980a9d520e2547b23137ace519d88894fa8f12c992b52e35e2fb89` |

All used host envelope 270,532,672 B and device headroom 536,870,912 B. The
representative shape covers unequal linear head counts, nonunit RoPE, top-8 and
multiwarp projections. These are correctness results, not measured performance.
[MODEL-GPU-PLAN.json](MODEL-GPU-PLAN.json) retains the reviewed pre-execution plan
and all immutable fixture/source/binary pins; its historical hold text is not a
current authorization. Host evidence directories (not GitHub downloads):

- `/tmp/kadan-112-tiny4-run-ks0jwed6`
- `/tmp/kadan-112-tiny8-run-jvcabr6g`
- `/tmp/kadan-112-representative4-run-724a4ycy`
- `/tmp/kadan-112-tiny40-run-yuxbms1_`

Each contains `result.json`, actual capture, process stdout/stderr and before/after
service/GPU/process snapshots. No retry or tolerance change occurred.

## Exact snapshot and numeric inputs

Installed snapshot: `small-1355db6a052410cfd62085d94b58866fd0f2c3c5`.
Host root: `/var/lib/docker/volumes/kadan_model_data/_data/small-1355db6a052410cfd62085d94b58866fd0f2c3c5`.
Container root: `/var/lib/kadan/models/small-1355db6a052410cfd62085d94b58866fd0f2c3c5`.

Bounded metadata only verified:

- `config.json`: `58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc`
- `generation_config.json`: `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`
- `tokenizer_config.json`: `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b`
- 40 text layers (30 linear, 10 full), 31,333 native bindings, H=2048,
  vocabulary=248320, 256 experts/top8, intermediate/shared width512.
- One input **248044**, confirmed BOS in architecture and generation JSON.
  It is also one EOS ID; this is an input, not an instruction to terminate before
  forwarding. Predicted EOS membership uses `{248046,248044}`.
- Batch1, context capacity1, position0, mask1, exactly one forward; no tokenizer,
  template, sampling, chat request or autoregressive generation loop.

Full shard hashes are intentionally **not available yet**: creating them reads
actual payloads and requires approval. Before reference execution, record SHA256
of every index-listed shard plus config/index/generation/tokenizer files, exact
file sizes and resolved paths. Recheck against that manifest before native
execution and after completion. Protect the snapshot from writers/downloads for
both runs. Read-only mounts protect those processes' view from their own writes;
they do not prevent another host process modifying backing files. Any detected
change invalidates all results. Stat checks are not cryptographic authentication.

## Corrected native envelope and discrepancy

Current metadata-only planner, capacity1, 256 MiB metadata quota:

```
layers=40 bound_items=31333 vocabulary=248320 arena_bytes=20897997312
minimum_staging_bytes=4100 metadata_used=40898511 eos=248046,248044
```

| Reservation | Bytes |
|---|---:|
| Native arena | 20,897,997,312 |
| Explicit device headroom | 536,870,912 |
| Required device envelope/free memory | **21,434,868,224** |
| Model host (256 MiB metadata + 1 MiB staging + 1 MiB control) | 270,532,608 |
| Host logits (248320 × 4) | 993,280 |
| Required native host envelope | **271,525,888** |

The old capacity-one estimate 20,897,995,392 was derived from capacity128 by
subtracting raw per-token state/scratch growth. In `full_attention.cpp`, the
probability scratch has `heads * capacity` floats. Here heads=16: capacity1
contributes 64 B, capacity128 contributes 8192 B. The fixed full-attention scratch
is 118,784 B (256-byte aligned). `decoder.cpp::plan` rounds the complete scratch
region up to 256 B: capacity1 adds **192 B padding**, capacity128 adds none.
Ten full-attention layers therefore require **10 × 192 = 1,920 B** more than the
linear extrapolation. KV regions remain aligned. The layout, not the estimate,
was always used by admission; this is a documentation arithmetic correction.

Current GPU0 UUID: `GPU-e30b6419-2c6d-f550-61d6-16166a920dac`.
Observed free 15,108,931,584 B is 6,325,936,640 B short. API PID78579 used9706MiB
on GPU0 and11020MiB on GPU1. Its removal suggests the test may fit, but post-unload
measurement is mandatory. Do not terminate this process or desktop clients.

The live Python GPU0 budget is20,014,117,683 B, **1,420,750,541 B below** the native
envelope even with no reservations. Never raise that budget or pretend the native
ledger acquired a Python lease. The operator must authorize a separate one-shot
native envelope under an exclusive maintenance window, independently verify RAM,
physical VRAM and other residents, then let native admission check again.
Require no active model download/replacement or other snapshot writer; a complete
model selection alone does not prove that no background writer exists.

## Reference capture path: CPU, bounded, separately reviewable

Use a new **test-only standalone capture wrapper**, not an HTTP completion or
`adapter.generate`. It is not implemented/pinned yet; its source, CPU fixture
tests and executable/container provenance are prerequisites to execution. This
plan must not be treated as an already runnable reference command.

The wrapper's exact contract is:

1. Use existing `api.inference.llm.qwen.build_qwen` with device `cpu`, an isolated
   `ResourceManager(host_bytes=42949672960, device_bytes={})`, and a temporary
   entry ID unique to this test. No imports of `api.server` or the runtime singleton.
   Force offline/local model access, no GPU visibility, two intra-op threads and
   one inter-op thread, `model.eval()` and `torch.inference_mode()`.
2. Set effective context1 using `configure_context(adapter, 1)`. The existing
   eager HF attention and existing Kadan weight-only projections remain unchanged.
   Explicitly disable optional kernel/hub substitution and verify the selected
   functions against reviewed installed source; fail if this cannot be established.
   Do not use `torch.compile`, a downloaded model implementation or remote code.
3. Hold the adapter host lease and a **512 MiB host request reservation**, then
   directly call its text model once with `input_ids=[[248044]]`, all-one mask,
   position0, `use_cache=True`, `return_dict=True`, `logits_to_keep=1`. Record fresh
   cache length1 and reject extra forwards. Standard HF first-token linear
   attention uses its prefix/chunk equations; native uses incremental recurrence.
   This independent path difference must be included in numerical review.
4. Validate exactly248320 finite BF16 logits, copy to FP32 containers without
   changing values, choose lowest-ID greedy argmax, derive predicted EOS from the
   pinned generation set. Capture header `[0x4b4d4331,1,248320,1]`, record
   `[248044,selected,eos,1]`, logits, DONE. Expected size **993,316 B**.
   Use exclusive creation, mode0600, fsync; publish DONE only after successful
   close and zero isolated reservations. Never derive expectations from native.
5. Retain diagnostic router IDs and per-layer output hashes for review, without
   extra forwards; tied router probabilities around top8 are a semantic review
   trigger because native lower-ID ordering and HF `topk` ties need not coincide.
   Drop output/cache references, close adapter, release request reservation,
   verify an empty isolated ledger, exit. Failed/timed-out captures are invalid.

Installed reference packages observed: torch2.14.1, transformers5.17.0,
safetensors0.8.0, accelerate1.10.1. Installed HF model source SHA256:
`42b6ca1cd9a2f0754c3dec97f0708dfa7e97e1b6edd9fac88f3139e4e36316c2`.
Pin the current API image ID, wrapper, Kadan reference modules and dependency
source hashes before execution; package version strings alone are insufficient.

Header-derived reference budget: all stored tensors23,407,580,856 B; conservative
non-expert doubled dense charge5,411,035,568 B; adapter host reservation
**28,885,725,288 B** including64MiB. With512MiB request reservation total is
29,422,596,200 B, inside40GiB. The request estimator at one token gives cache
66,887,680 B + workspace268,714,256 B + host1,048,608 B =336,650,544 B, below512MiB.
The existing CPU cache admits one largest packed entry at a time and does not
materialize the full dequantized model. No GPU allocations are allowed. Cgroup
40GiB includes page-cache/allocator/framework use; the ledger is not an RSS limit.
OOM or budget exhaustion fails the experiment, never increases limits automatically.

Proposed isolated invocation after wrapper/image review (placeholders must be
resolved and pinned; not commands to run now): CPU-only container using the
existing image ID, `--network none --cpus 2 --memory 40g --memory-swap 40g`, no
`--gpus`, `CUDA_VISIBLE_DEVICES=''`, offline flags, `--read-only --cap-drop ALL`,
read-only code and checkpoint mounts, and one fresh writable artifact directory.
Entrypoint is the reviewed wrapper via GNU timeout, **1800s + 5s TERM/KILL grace**;
host supervisor deadline1815s+5s. Record its container ID first. A timeout of the
Docker client is not container cleanup: inspect and, only for this disposable
reference container, terminate it if still running; verify exit and host-memory
release. Never stop the API container. Keep failure artifacts for review.

Reference CPU capture can precede the GPU maintenance window after explicit
payload/reference authorization and snapshot-writer exclusion. Existing service
and reference host charges fit observed host admission, but fresh physical/cgroup
headroom >=40GiB remains mandatory. This avoids holding the API offline for the
CPU reference. No duration is measured yet: two-thread CPU work over the selected
experts and a large vocabulary head may take minutes. Thirty minutes is a fixed
resource ceiling, not a prediction or benchmark target.

## Comparison criterion, fixed before seeing actual logits

Initial acceptance is **ABS_TOL=0, REL_TOL=0**, all logits finite, exact input,
selected token, predicted EOS and progress, and complete artifacts. This is a
conservative exact-agreement test, not an assertion that independent BF16
implementations are guaranteed to agree. A mismatch is a diagnostic stop, not
permission to widen tolerances and not by itself proof of an algorithm bug.

There is no defensible nonzero global tolerance from the available evidence.
BF16 rounding has unit roundoff2^-8; sequential FP32 summation over2048 products
has a first-order gamma bound about1.22e-4 times sum(abs(products)), **not** the
relative output. Cancellation, RMS normalization and40 nonlinear layers prevent
turning that local bound into a useful whole-model logit bound without data.
Top8 routing is discontinuous near ties: even small arithmetic errors can change
experts, so a multiple of BF16 epsilon is not an independent guarantee. FMA/warp
reductions and HF chunk recurrence also differ. The four exact synthetic passes
support this first strict test, but do not establish actual-checkpoint bounds.

If strict comparison fails, retain both captures and router/layer diagnostics and
stop. Any later nonzero acceptance threshold needs a separate independently
reviewed error analysis/reference study and new authorization, never post-hoc
fitting to the observed maximum. The actual selected token and EOS flag remain
unknown until the independent reference is captured and reviewed.

## Queue/admission coordination: correction and unresolved review item

The earlier claim that a newly queued chat request reloads an unloaded model was
incorrect. At `5af05a673a09d009f1fd87bd38b87ca0cb471346`, the verified path is
`MemoryManager._execute` -> `LLMFeature.__call__` -> `RuntimeManager.complete`.
That path does **not** call `LLMFeature.load`. `complete` rejects unless state is
ready and the adapter exists; a request without a selected model can reject even
earlier in `LLMFeature.__call__`. After successful unload, queued chat does not
by itself reconstruct the LLM.

Remaining concerns are narrower: an explicit lifecycle load can reconstruct the
LLM, and other modality requests may allocate shared GPU resources. An unrelated
`flock` does not bind those producers; the native ledger is still separate from
the service ledger. Queue/admission observations remain useful evidence, not an
atomic cross-process reservation. Do not take, unlink or replace the service's
inference lock or write/cancel/delete Redis jobs for this experiment.

The prior nft/namespace firewall and socket-killing proposal is **withdrawn from
this plan**. It would expand the requested LLM-only window to ordinary API
availability and requires separate security-sensitive approval. No such gate was
implemented or applied; its commands have been removed. Do not install a firewall,
close connections or change network settings under this plan.

No external atomic maintenance lease currently exists. A quiet operator window
is not equivalent to such a lease and cannot bind concurrent producers. The
preferred eventual solution is an application-level maintenance lease, requiring
separately reviewed Python/API changes and explicit deployment approval. It is
not implemented or authorized by this plan.

The runtime reviewer is evaluating safer supported coordination. Its mechanism,
operator responsibilities and failure behavior must be reviewed before any
actual-model execution. No replacement implementation is proposed here. Pending
that decision, preserve the read-only checks: queue pending/unfinished counts,
active HTTP requests, video jobs, shared active leases/exclusive owner, explicit
lifecycle activity, physical memory and process residency. If the agreed
coordination cannot exclude conflicting lifecycle or other-modality allocation,
stop; do not silently substitute a broader operational gate or racy polling.

## Supported unload, native run and restoration sequence

The following are **proposed future commands**, not authorization. `RUN` is a
new operator-owned0700 artifact directory. Verify the API container identity and
service epoch before mutation. All commands/logs use the same operator session.
Execution also depends on the unresolved supported coordination described above.

```sh
# Read/save state before mutation; GET is harmless.
curl --fail-with-body --silent --show-error \
  --max-time 10 http://127.0.0.1:8000/model-lifecycle > "$RUN/lifecycle-before.json"
# Exactly one authorized unload, no body.
curl --fail-with-body --silent --show-error \
  --max-time 120 -X POST http://127.0.0.1:8000/model-lifecycle/unload \
  > "$RUN/unload.json"
```

An HTTP timeout is **uncertain**, not successful cleanup. Do not repeat unload;
poll GET and inspect residency. Require state unloaded, model_id null, all LLM
reservations/leases gone, queue empty and physical GPU0 free >=21,434,868,224 B.
Record the post-unload API CUDA context baseline (context may remain resident).
The native process must fit alongside this baseline and any other unchanged
resident. Shared native admission still checks physical free again. No service
budget enlargement or GPU1 use is allowed. Save original ready/small state,
context65,536, max_output256, service epoch, budgets and snapshot selection.

Before native: the independent CPU reference must already have passed capture
validation and review, with its chosen token/EOS and hash added to the approved
run manifest. Recheck all native/reference/metadata/shard hashes, queue and the
separately reviewed coordination conditions. Planned exact native command (only RUN varies as a fresh path):

```sh
sudo -n timeout --signal=TERM --kill-after=5s 915s \
  /tmp/kadan-stack-cuda/kadan-model-correctness --execute \
  /var/lib/docker/volumes/kadan_model_data/_data/small-1355db6a052410cfd62085d94b58866fd0f2c3c5 \
  0 1 271525888 21434868224 536870912 900 "$RUN/native.capture" 248044
```

Native binary SHA256 remains
`0c2e7fef0f7b2911b83d7ab6c68415049436d33268d6d27b391907be0d02331c`;
comparator `/tmp/kadan-stack-cpu/kadan-model-compare` SHA256
`656e6ee654e1cd1ab7b606d552c9f6e27d002ced32c74aac1b9f20d405a635bd`.
These binaries are unchanged by documentation commits. Re-review any rebuild
that changes a hash. One arena allocation, one model ticket, separate host logits
ticket, and **2,314 static forward launches**; no generation loop. Internal
watchdog900s, external915s+5s kill grace. Loader admission/validation scans about
20.8GB of selected text payload, mostly before compute; the900s ceiling allows
bounded loading/checking without assuming warm disk cache or GPU throughput.
It is not an expected completion time and is never extended after a failure.

Require exit0, 993,316 bytes, correct header, one committed input248044, finite
logits, reference-selected ID/EOS, DONE and zero final reservations. Compare:

```sh
/tmp/kadan-stack-cpu/kadan-model-compare "$RUN/reference.capture" "$RUN/native.capture" 0 0
```

Always record actual process/timeout status, before/after source/fixture/shard
hashes and GPU process residency, including on a failed run. Require the native
PID gone, no new compute process, and GPU0 allocation returned to the saved
post-unload baseline within telemetry granularity. Allow up to30s observation,
not another inference. Timeout/KILL alone proves neither `cudaFree` nor device
context cleanup. An uncertain process, retained device bytes, failed telemetry
or nonzero native reservation prevents overlapping any reload with native state.
Do not reset a GPU, kill a production process, rerun or widen tolerances.

After **observed** native cleanup, restoration is required even if numerical
comparison failed; the one-shot restoration must be included in approval:

```sh
# Exactly one no-body load restores saved model/context, not new settings.
curl --fail-with-body --silent --show-error \
  --max-time 30 -X POST http://127.0.0.1:8000/model-lifecycle/load \
  > "$RUN/reload-accepted.json"
# Poll GET via the same endpoint; HTTP202 is acceptance, not readiness.
```

Poll up to900s for ready/small, configured/effective context65,536, output256,
unchanged service epoch and budgets, no errors/active work. Check read-only
health/metrics, GPU residency and Redis queue. Reload can have different cold
expert-cache occupancy; do not demand byte equality with the pre-unload warm
cache, and do not generate a token to warm or test it. Unload discards model
residency and warm/prefix caches but preserves saved selection/context and stored
outputs. No container restart is needed. Reload reads actual payloads and must
be covered by the window's authorization.

Verify the ordinary host/frontend read-only health path and the restoration
outcome. If reload fails or times out, do not repeat or restart; record current
state and report incomplete restoration. If native cleanup is uncertain, do not
issue the planned reload: notify the operator/reviewer and retain the agreed
coordination conditions. There is no automatic reload on queued chat and no
network gate to remove under this corrected plan.

## Duration, service impact and remaining review gates

Reference may be prepared before outage under its separate30-minute CPU ceiling.
For the maintenance window, propose maximum queue drain60s, unload120s, each
pre/post full-shard hash pass600s, native915+5s, cleanup observation30s,
reload900s. These are **escalation and observation deadlines**, not an additive
worst-case duration or a guarantee of cleanup/restoration. Cancellation, device
work, process termination and reload can remain uncertain beyond those deadlines;
there is no defensible upper bound on service outage from the present evidence.
The previous approximately54-minute worst-case claim is withdrawn. Hashing is sequential on one CPU; it reads the
full23.4GB checkpoint per pass. Do not benchmark disk/GPU to refine this estimate
without permission. If a hash pass exceeds600s, abort the experiment and restore
service when safe; do not silently expand the window. Reviewers may choose
smaller ceilings before execution. The requested window makes the selected LLM
unavailable from unload until successful reload; queued chat rejects while it is
unloaded. It does not itself disable all API access. Any additional impact from
the still-unresolved coordination mechanism must be reviewed and approved, not
assumed to be part of an LLM-only window.

Still required before executing anything in this plan:

1. Andrew's #112 merge decision and explicit reference payload/inference plus
   maintenance window authorization. No merge is performed by this document.
2. Implement and independently review the test-only CPU reference capture wrapper,
   fixture tests, source/image hashes, numerical backend selection and container
   cleanup supervisor. It does not exist yet; do not invent its CLI/hash.
3. Independent review of initial0/0 acceptance and the one-BOS semantics; reference
   capture must be reviewed/pinned before the native command is released.
4. Resolve and review safer supported coordination for explicit lifecycle loads
   and other-modality admission. The broad network/socket gate is withdrawn;
   no alternative implementation or operational mutation is authorized here.
   Finalize the bounded supervisor/restoration procedure only after that review.
5. Approve checkpoint payload hashing, snapshot-writer exclusion, corrected native
   envelope, CPU reference40GiB cap and all time ceilings. Recheck physical and
   cgroup availability at each phase. No actual weight work is currently cleared.

All failures are fail-stop: preserve evidence, do not repeat a GPU/reference
stage or modify tolerance/budgets, restore service only after cleanup is known,
and report any incomplete restoration explicitly.
