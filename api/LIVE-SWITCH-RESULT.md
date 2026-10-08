# MR119 reviewed GPU1 acceptance — PASSED

Tested source: **101bc45b880e3467105cbbf3bf268fa227eb2a54**. MR119 remains draft/unmerged.
Executed 2026-10-08 19:22:03–19:29:09 UTC; host supervisor finished19:29:14 UTC.
Container exit0, OOMKilled=false, no abort reason. This is one real checkpoint
acceptance, not a broad performance/quality guarantee.

## Ordered switching and application RAM reuse

The real Redis FIFO completed **text A (`4`) → image B → text C (`7`)**.
Text parked before image execution; image parked before C; the same native child
PID20/session250c615d98ba1c922378a706d8b08733 served both text requests.

| Application cache counter | At park | After C |
| --- | ---: | ---: |
| Capacity | 20,825,156,824 | 20,825,156,824 |
| Actual retained payload bytes | 20,825,156,824 | 20,825,156,824 |
| Registered entries | 124,116 | 124,116 |
| Hits | 4,225 | 132,566 |
| Misses | 124,116 | 124,116 |
| Hit bytes | 2,444,388,432 | 23,269,551,224 |
| Checkpoint payload source bytes | 20,825,156,824 | 20,825,156,824 |
| Evictions | 0 | 0 |

Full packed weight occupancy, increasing application hits and **zero source-read
increment** prove application buffer reuse independently of Linux page-cache
hits. Initial child rchar20,886,279,216; reload rchar increment only1,423 bytes of
protocol/control activity. Parked RSS20,526,080 KiB. No SSD traffic claim is needed
for this proof, and no new disk cache/spill implementation is claimed.

Initial native load64.332s (previous94.584s). C start/restore49.239s, then about
0.53s to final text completion. RAM reuse works but restore remains slow; the
current loader still performs host validation and many uploads. That attribution
is from source inspection, not a separate native CPU profile. Warm text decode
throughput was not benchmarked by these one-token answers.

The runtime outer deadline is an **aggregate bound**: generation allowance plus
three planner/load/start deadlines, including warm requests. With load300s and
generation300s it is1,200s, not a separate strict300s generation deadline. Token
IPC remained30s. Command deadlines and the independent run/image supervisor caps
remain active.

## Image timing and unchanged output

Same exact Qwen-Image-2.1 revisiond26bb61231c349cf6b7896fa83353113880e1ba3,
Diffusers commit8d3c30bfda9b511c00992f40cff4170a5502814d, GPU1 RTX3090,
BF16,2048×2048,40 steps, seed42, same prompt/guidance/VAE tiling. Only offload mode
changed from sequential leaves to text_encoder→transformer→vae component offload.

**Forward292.501s versus498.965s previously: observed41.38% reduction (1.706×).**
This is still about4m53s per image and is not interactive latency.

| Measured new phase | Seconds |
| --- | ---: |
| Host pipeline construction, outside forward | 3.118 |
| Prompt encoding | 3.971536 |
| Transformer calls, all40 | 273.897643 |
| First transformer, includes transition and profiler | 17.270289 |
| Remaining39 transformer calls | 256.627354 |
| Steady transformer median | 6.594313 |
| Steady transformer range | 6.463652–6.613765 |
| VAE decode including outgoing-transformer transition | 12.842227 |
| Image postprocess | 0.107588 |
| Forward residual, including trace export/scheduler/hooks/instrumentation | 1.681009 |
| Forward return to FIFO publication completion | 1.429 |
| Complete image FIFO request | 297.108 |

Actual tile counter: **121 VAE decoder forwards**, with VAE retained for its decode
phase. Earlier22.922s after-last-step interval included VAE, postprocess and hook
cleanup, so it is not a clean old VAE timing. Earlier139.912s first callback included
prompt/startup/first step, so it cannot retrospectively be labeled prompt-only.

Previous steady callback median8.618s versus new transformer median6.594s gives
about23.48% reduction; callback bookkeeping is small but these are not precisely
the same timer boundary. OS checkpoint cache is warm in the new run, explaining
part of the large construction/first-stage improvement. The41.38% total is an
observed same-output comparison, not a cache-controlled causal estimate of the
offload change alone. Profiling synchronizes phase boundaries; first-call tracing
and1.28s trace export add overhead. No quality settings were silently reduced.

Output `evidence/media/images/0e9c06a7-8d30-4c10-bc00-df08be3a65cd/0.png` is
**byte-for-byte identical** to the previous run's visually verified apple PNG:
SHA256 `6df59d86af4143efd1e3e5b419812754bd2b9ed1b10ccd33c3bb2ecac4dbf42b`.

### Transfer and remaining compute bottleneck

The first-transformer Chrome trace is `evidence/transformer-trace.json` (~9.96MiB).
It includes offloading the text encoder and loading the transformer:

- GPU DtoH pageable memcpy events:8.145995s,756 events.
- GPU HtoD pageable memcpy events:1.558201s,306 events.
- Dominant BF16 CUTLASS matrix kernel:3.5843s,224 calls.
- CUDA flash-attention kernel:2.3099s,32 calls.

These are summed event durations, not disjoint CPU wall buckets; do not sum parent
operators with their child memcpy/kernel events or extrapolate transition copies
across40 steps. The transformer remains GPU-resident after the first transition.
The trace confirms GPU BF16 matrix and flash-attention execution; no unexpected
CPU matrix execution was identified. It is not an all-phases CPU operator audit.
At unchanged settings, steady denoising is now substantially compute-bound:
roughly5.9s of dominant first-step matrix/attention device work is consistent with
the6.59s steady calls. Further large gains likely need deeper kernel work or a
separately evaluated resolution/step/precision tradeoff, not just another offload
policy. No such quality tradeoff was applied.

## Measured memory and cleanup

Logical host transition envelope97,795,899,032 bytes (91.080GiB) within96GiB;
component image device reservation22,548,578,304 bytes (21GiB), plus two512MiB
persistent contexts within22GiB parent capacity. Component source files are
conservative sizing inputs; actual loaded parameter bytes were BF16:
text encoder17,534,247,392; transformer14,230,249,472; VAE675,480,808.

| Torch phase including outgoing component at handoff | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| Prompt encoding | 17,588,168,192 | 17,779,654,656 |
| First transformer transition | 17,558,656,512 | 17,779,654,656 |
| Later transformer | 16,689,225,216 | 17,335,058,432 |
| VAE including outgoing transformer | 14,266,277,376 | 17,335,058,432 |
| Postprocess | 816,039,936 | 1,342,177,280 |

Torch phase peaks exclude native context and external desktop allocations; the
physical monitor includes them. Maximum sampled GPU1 usage during image18,413MiB;
whole-run maximum22,737MiB occurred in text residency. Minimum physical GPU1 free
1,380MiB. Cgroup peak55,748,202,496 bytes (51.920GiB). Minimum host available343.249GiB.
CPU maximum71.75°C; GPU1 maximum82°C. No thermal/power-brake flag or guard event.
Five-second physical sampling cannot prove an instantaneous peak bound.

Final cleanup confirmed all model reservations released, leaving only Python's
framework context until process exit. Test process exited; no owned native child
remained. GPU1 returned to862MiB. GPU0 stayed21,779MiB total, live native PID1452475
stayed21,508MiB. Live API remained healthy and GET /health returnedstatus ok.
Original checkout HEADb6a6c901 and its six local untracked deployment files remain
unchanged. Test containers/Redis/network were removed; evidence and binary retained.

## Reproducibility and CI

Source101bc45b880e3467105cbbf3bf268fa227eb2a54; native sources unchanged since
2fab5f169a4d9d9ca300bfeb6d3eb50c81fd8cb4. Rebuilt only native target on cached toolchain.
Worker SHA25602309c0817596588b52f86586748fe4120ce81c3b652607fe6a27016e2496e58.
Runtime image sha256:eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa.
See manifest.json, harness-SHA256SUMS, live_switch.py, monitor.py, source/, native/,
and evidence/{events.jsonl,physical.jsonl,result.json,supervisor-result.json,
container.log,compute-before.txt,compute-after.txt,container-isolation.txt}.

All final-head CI checks passed before execution. API run37830803264:425 tests,
2 skips, plus container imports and generated frontend checks. Native run37830803235
passed. Push equivalents37830794948/37830794855 also passed. MR119 remains draft and
unmerged; component mode remains opt-in. No production deployment occurred.
