# Real-weight denoiser review slice (not executed)

This separate draft stacks on MR121; it does not change API imports/configuration,
production Python orchestration, UI, native text or the synthetic probe. The
synthetic SHM timings establish neither trained-weight parity nor image latency.
No GPU runs/model loads are authorized by merely importing these files.

## Concrete validation ladder

1. `capture.py` loads the exact local d26bb61231c349cf6b7896fa83353113880e1ba3
   checkpoint with the pinned pipeline, BF16, component CPU offload, seed42,
   2048²,40 configured steps and the same red-apple prompt. It executes unchanged
   eager prefill and captures the first cached timestep. Each block's actual
   weights, incoming/outgoing hidden state, global-position RoPE, target/t=0
   modulation, compact post-RoPE prefix and exact optional key mask are saved.
   Final adaptive norm parameters/conditioning and output projection/expected
   output are captured too. It stops after that projection, before full denoising
   or VAE; this is not a40-step image result. No guidance/mask/processor changes.
   Digests, timestep, scheduler config and pinned revision accompany the capture.
2. `replay.py` validates artifact digests and bit-exact sequence/head exchange
   round-trips independently of arithmetic tolerance. Trained BF16 checkpoint
   weights/prefix are cast to FP32, and128 target rows retain their captured global
   RoPE positions. These shortened arithmetic comparisons use atol=rtol=2e-5;
   they are not full-sequence FP32 attention or a newly generated FP32 prefill.
3. At the actual16384 target rows, all32 blocks are compared independently on
   identical captured eager inputs in BF16. Both ordinary eager replay and
   Ulysses must match the captured reference. Each comparison is persisted before
   its acceptance gate; failure stops both ranks and the entire ladder.
4. Free-running1/4/32-block chains feed each computed result into the next block,
   never reset hidden states from the teacher trace. Each layer is checked; the
   32-block gate includes final adaptive norm and projection. Single-GPU rank0
   and two-GPU Ulysses use the same weights, prefix, masks and conditioning.
5. Only after those gates pass, repeat the full32-block cached core plus final
   norm/projection with2 warmups and5 measured repeats. Include initial sharding,
   per-block validation/consensus, QKV exchange/layout, inverse exchange/layout,
   residual/MLP and final target-output gather in the Ulysses timing. The single
   GPU baseline executes only on rank0; rank1 is idle (but its replica still
   occupies memory). Preparation, disk I/O and streamed numerical audits are
   outside the latency window and reported separately. Every repeat is checked.

The common BF16 gate remains atol=rtol=.02, including chains: **no tolerance
widening for accumulated drift**. Reports include violation/nonfinite counts,
maximum normalized tolerance ratio, absolute error, RMSE, relative L2, and signed
reference/actual plus reference magnitude at the largest absolute error. A0.03125
error alone is not a one-ULP claim. Metrics transfer/compare at most262144 elements
per CPU chunk; they do not retain32 full hidden states on GPU. Per-rank local
metrics remain separate and must be aggregated appropriately, not averaged to
hide a rank's violations. FP32 reports use the FP32 bound.

## Ownership, memory and deadlines

`launch.py` defaults to a read-only plan. It is derived from the reviewed MR121
launcher and retains exact-container stop/start, idle-queue admission, desktop
PID/starttime baseline, unknown-owner rejection, physical return checks, unchanged
source/config checks, model-ready recovery and no unrelated-service recreation.
`supervisor.py` holds the existing model-volume inference.lock inode and supervises
one capture process or both replay ranks. There is no secondary live consumer.
No DB writes, model downloads, compilation or whole-image builds are used.

These **new bounds require independent review before execution**:

- Capture and replay each have900s supervisor /930s host limits. NCCL/control
  collectives have120s timeouts. Worst-case pause envelope is35 minutes including
  recovery; this is a ceiling, not an expected measured duration.
- Container hard RAM/no-swap128 GiB,4 CPU,1 GiB SHM. Require160 GiB host available
  before pause; abort below16 GiB. Model construction, duplicate CPU tensors,
  mmap/file cache, comparisons and both rank processes share that128 GiB ceiling.
- Replicated transformer parameters cost about13.25 GiB **per GPU**, not pooled.
  Each PyTorch allocator is hard-limited to22 GiB; require22 GiB physical free
  per card after stopping API. Physical free256 MiB/GPU90 C/CPU80 C guards remain.
  NCCL/context allocations outside the allocator are watched physically.
-32 GiB disk capture budget with pre-write estimates and a host growth guard;
  require64 GiB free. Expected roughly22 GiB for weights and32 CPU input/output
  pairs plus metadata/tail. Disk is a slow artifact tier, not extra fast RAM.
  Partial files count against the budget and are retained for diagnosis.
- Replay loads each reference packet from disk on demand. Only current hidden
  output is retained on GPU. The32 resident block replicas, per-layer prefix/
  RoPE/modulation and small tail are explicit GPU residents. Both full and
  head-sharded prefix copies are retained for comparison and counted; memory
  numbers are not an optimized production-cache footprint.
- Both GPUs' allocated/reserved peaks and sampled physical memory, utilization,
  memory-controller utilization and power are recorded; samples may miss short
  instantaneous peaks. Container memory.current is sampled, including charged
  file cache. Capture is one-GPU execution; replay is two-rank execution.

## Status and remaining full-denoising gate

CPU-only contract tests passed4/4 in the pinned image; module compilation and
read-only launch plan passed. These tests cover metric thresholds/streaming,
nonfinite rejection, meta-module state loading, preservation of shortened global
RoPE/mask shape and chain-vs-reset distinction. They are not real-weight or CUDA
acceptance. No live pause or checkpoint execution has occurred for this slice.

After this first captured timestep passes, a subsequent reviewed extension must
check selected early/middle/late teacher-forced timesteps and full free-running
latent trajectories using identical prefill/cache/scheduler/seed. Then compare
final tensors/pixels and whole image wall time. This slice deliberately cannot
claim40-step parity, full editing/reference-image correctness or a production
speedup. It must not enable the API's production split. MR121 stays draft.
