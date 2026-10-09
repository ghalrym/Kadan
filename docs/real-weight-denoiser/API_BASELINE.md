# Isolated A/B baseline runner for API validation

`launch_api_baseline.py` defaults to a read-only plan. It does not activate the application split. Execution requires a clean exact source commit, successful exact-head push API/native CI, independent case/source/settings review, an idle Redis FIFO, cool admission and no unexpected GPU owners. The original API container is stopped and retained intact, then the exact captured ID/config is restored after physical cleanup. Other service IDs are checked. Uncertain cleanup leaves the API stopped with a quarantine error.

The new runner imports established host identity, thermal, driver ownership and CI primitives from `launch_trajectory.py`; the frozen trajectory harness and application inference code are unchanged. A single isolated eager pipeline executes A or B in a fresh process. No Ulysses adapter, persistent pair, shared request cache or saved trajectory tensors participate. Both cases use the pinned Qwen revision, explicit 2048²/40/BF16/true_cfg_scale=1/KV-cache settings, CPU generator, component offload and VAE tiling. Scheduler configuration, source hashes, precision/runtime/device identity, load time and generation time are retained. These baseline times are not normal API latency.

Limits: 128 GiB container RAM, no swap, two CPU quota, one intra/inter-op thread, 1 GiB verified SHM, 22 GiB Torch allocator, 900-second stage; 1,800-second overall pause including a reserved 600 seconds for recovery. Require 160 GiB host available and 64 GiB disk headroom before pause. CPU admission <60 C for five samples, runtime CPU <80 C/GPU <90 C, unchanged from the existing diagnostic protocol. The baseline allocator profile is intentionally the established eager 22 GiB profile, not the API rank's 20 GiB profile. Only physical GPU `GPU-e30b6419-2c6d-f550-61d6-16166a920dac` is exposed and verified inside the child. There are no collectives in this single-process oracle.

The supervisor opens the existing `/models/inference.lock` inode read-only and holds its exclusive lock until child teardown. Read-only model volume is reused; no copied weights, image rebuild or new model download. A and B require separate launcher invocations/windows, restoring the original API between them.

Example **after independent clearance**, substituting the reviewed SHA and complete retained-root list:

```sh
python3 docs/real-weight-denoiser/launch_api_baseline.py
python3 docs/real-weight-denoiser/launch_api_baseline.py \
  --execute-reviewed "$reviewed_commit" --case A \
  --review-record /absolute/path/review-A.json \
  --evidence /absolute/path/new-baseline-A \
  --retained-roots /absolute/path/existing-capture /absolute/path/prior-run-1 /absolute/path/prior-run-2
```

The example roots are placeholders, not a complete ledger. Include **all** preserved capture/holdout/replay/failed-v1/v2 reference/repeat/candidate directories and later API validation outputs. Roots must be distinct/non-overlapping; never delete history to fit. The launcher enforces the established 32 GiB global ceiling plus a 64 MiB per-window evidence cap (PNG <=24 MiB, bounded logs/metadata). For B also include A's evidence root. The broader API validation plan still caps all new artifacts at 512 MiB.

Review JSON binds `protocol="api-isolated-image-baseline-v1"`, exact `source_commit`, `case="A"` or `"B"`, `settings` from `api_baseline.settings(case)`, `criteria` from `trajectory_contracts.CRITERIA`, `decision="approved-for-bounded-execution"`, and actual independent `reviewer`/`review_reference`. The tool does not create or fabricate approval records.

Outputs are in `trajectory-evidence/baseline/`: immutable `output.png` and `manifest.json`. The enclosing evidence includes CI/review, bounded stderr/logs, thermal readings including rejected samples, GPU/cgroup observations, supervisor/queue-inode proof, exact container/config digest and restoration outcome. Failed runs retain failure evidence and cannot pass baseline verification. Generation verifies all 40 transformer/scheduler pairs and raw VAE finiteness before normalization/clipping. PNG publication never overwrites an existing baseline.

Offline comparison needs no GPU:

```sh
python3 docs/real-weight-denoiser/api_baseline.py --case A \
  --commit "$reviewed_commit" --output /absolute/path/new-baseline-A/trajectory-evidence/baseline \
  --compare /absolute/path/api-output.png
```

It verifies baseline identity/hash and RGBA dimensions, then applies the existing max-channel <=2, RGB mean <=0.5 and alpha mean <=0.5 uint8 thresholds. It reports pixel equality and hashes. A B output must be compared with B, never A. This establishes final-output parity; no B intermediate-trajectory claim is made.
