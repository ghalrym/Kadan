# Reviewed synthetic two-GPU result

Executed5074ea50e95a51b9b40d684d16903e194db41b03 on2026-10-08 after independent
source review and all exact-head PR/push checks passed. PR CI: API37846949602,
native37846949598. MR121 remains draft. No production dual-GPU execution enabled.

The prior CPU loopback smoke proved only process startup/Gloo. The measurements
below are from two actual RTX3090 CUDA ranks, Torch2.14.1+cu130/NCCL2.30.7, with
PHB topology and peer access false in both directions. NCCL channel logs explicitly
report `via SHM/direct`; socket/loopback plugin initialization is also logged, but
must not be mistaken for the selected intra-node data-channel transport. This is
host shared-memory transport, not direct GPU P2P or NVLink. No driver, IOMMU or
security settings were changed.

## Communication

BF16 batch1,32 heads,128 dimensions/head. Each mode had3 warmups and10 measured
iterations. Reported times include packing/reorder, synchronization and collective
execution; effective rates use logical off-rank bytes, not measured PCIe traffic.
The table uses the slower rank's median. One short sweep is not a statistical
performance guarantee or an end-to-end model benchmark.

| Local rows/rank | Target K/V gather | Ulysses QKV + inverse | TP two all-reduces |
| ---: | ---: | ---: | ---: |
| 1024 | 2.165 ms | 2.122 ms | 3.899 ms |
| 4096 | 8.949 ms | 8.290 ms | 15.860 ms |
| 8192 | 17.983 ms | 16.907 ms | 32.039 ms |

At8192 rows/rank (16384 global target rows), gather/Ulysses logical payload is
128 MiB/rank; TP is256 MiB/rank. Effective rates are6.95,7.39 and7.80 GiB/s
respectively. Ulysses was about6% faster than gather in this sweep, while TP's
larger payload took about1.9x Ulysses time. This alone does not select the full
executor: TP's lower replicated-weight cost and actual GEMM/attention time still
need comparison. Multiplying communication medians by32 blocks gives roughly
0.575/0.541/1.025 seconds per step, an arithmetic estimate that excludes compute,
cache handling, control consensus, model transitions and overlap.

Communication peak allocated bytes/rank at the largest shape were1,015,152,640
(gather),1,149,370,368 (Ulysses),344,064,000 (TP). Peak reserved reached
1,633,681,408 bytes/rank; allocator caching carries across modes, so TP's reported
reserved bytes are not its isolated requirement. GPU physical sampled peaks were
2040/2606 MiB including contexts and GPU1 desktop baseline. Samples are about1s
apart and are not a guarantee of instantaneous maxima. CPU peaked68.75 C, GPU
65/64 C; guards passed. No full transformer weights were loaded.

## CUDA parity and its limits

Both gather and Ulysses passed16 cases/rank against the pinned eager cached block:
FP32/BF16, even/odd target lengths, masked keys/padding, compact immutable prefix,
global RoPE and target versus t=0 modulation. One prefix includes an eager
condition-image segment. Real projection width4096/32 heads/128 depth was tested
with27 prefix and64 target rows; large16384-token communication is a separate
payload test, not large-sequence block parity. Weights and inputs were synthetic.

Maximum FP32 absolute error was1.6689300537109375e-6 (tolerance atol=rtol=2e-5).
Maximum BF16 absolute error was0.03125, RMSE0.0015414860 in the real-width case.
BF16 passed **combined** atol=rtol=.02;0.03125 is not below the absolute tolerance
alone. Do not call the result bit-identical or full-model image equivalence.
Prefix K/V BF16 storage at real width was442368 bytes/rank for gather and221184
for Ulysses. Each prefix remained unchanged. FP32/BF16 CPU initialization took
about1.06/1.24s in the real-width cases. Rank-reported numerical/communication
work took5.119s, while normal supervisor lifetime including startup was25.165s.

Still unvalidated: distributed global prefill, full32-block denoiser, trained
weight parity across40 steps, final pixels, arbitrary editing/reference layouts,
end-to-end latency, model parking/transitions and shared production two-device
reservations. No native LLM TP claim follows from these image-block results.

## Failure handling and exact service restoration

Peer-exit injection returned rank1 exit71; torchrun stopped rank0 with SIGTERM.
Injected OOM returned rank1 exit72 with an OutOfMemoryError record; torchrun again
stopped rank0. Supervisor lifetimes18.450/18.697s include startup, not just failure
reaction time. Both containers exited1, had no host OOM and reaped torchrun.
OOM was an injected exception after NCCL setup, not deliberate physical exhaustion.
GPU process identity and physical-memory return checks passed after every case;
recognized desktop processes were preserved. All cases stayed within bounds.

The launcher exited0 and restored the same API container
1efcf5729f776dd90bbc0611c63a474bc696cadc0d48b9011fbb320e167b6499, image
sha256:eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa.
Restarted21:33:33.947302404Z; HTTP/Docker health and model state ready verified.
Normalized mount/config identity matched, and frontend/Redis/Postgres IDs were
unchanged. Runtime source is still MR119 dea7b0f with single-GPU native text.

Machine-readable evidence and selected transport/error logs are in
evidence-5074ea5/. Complete local samples/inspect/logs are retained at
task-4/dual-gpu-reviewed-5074ea5. The next small change should evaluate isolated
cached denoiser compute with real weights and exact reservations, before any API
integration or production activation.
