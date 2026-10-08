# Regional compilation candidate: failed GPU fit

Run2026-10-08 19:54:25–20:02:01UTC, isolated physicalGPU1. Source6e9529d;
exact harness/manifest published at15ad0e2. Existing GPU0 native PID1452475 stayed
at21,508MiB; physical GPU0 remained21,779MiB and live /health returned ok afterward.

## Completed evidence

A returned4; eager component image B completed in291.671s; C returned7 after
restoring the same native child from its full20,825,156,824-byte application cache.
Source-byte counter stayed unchanged and cache hits rose4,225→132,566 with no
evictions. This repeats the accepted text/image/text behavior.

B kept2048×2048, BF16,40steps,seed42 and the exact apple prompt. PNG SHA256:
`6df59d86af4143efd1e3e5b419812754bd2b9ed1b10ccd33c3bb2ecac4dbf42b`.
It is byte-identical to both earlier completed component/sequential outputs.
Prompt3.089s; VAE12.559s; postprocess0.112s. Unprofiled cached transformer median
6.602s(excluding call1 and profiled call10). No precision or quality setting changed.

Cached-step10 trace from the preceding identical eager run shows224 dominant
CUTLASS BF16 matrix kernels totaling3.572246s and32 CUDA flash-attention kernels
totaling2.376513s. Main matrix shapes are[16384,4096]×[4096,4096](128calls),
[16384,4096]×[4096,12288](64calls), and[16384,12288]×[12288,4096](32calls).
Attention Q is[1,32,16384,128], K/V[1,32,16411,128]. Host/device copy kernels
are only tens of microseconds here; conversion/device-copy operators are distinct
from model offload transfers. These are device durations, not additive CPU timings;
do not double-count parent SDPA/linear operators. Eager is already flash-backed.

## Failed candidate and allocation evidence

D enabled compile_repeated_blocks(default,fullgraph=True), preserving the current
attention processor. One133-node backend compilation completed in8.578017s.
That is one completed backend call, not total compile cost for a full image.
During transformer call1, the supervisor sampled GPU free memory147MiB, below the
256MiB guard, and requested cooperative cancellation. Torch recorded peak allocated
23,396,391,936bytes and reserved23,601,348,608bytes, above the image phase envelope.
D produced no image. E/F/G did not run. There is no valid compiled speedup, numerical
quality comparison, steady-step trace, or warm park/unpark/recompile result.

Pinned Diffusers transformer_qwenimage21.py explicitly clones the27-token text
K/V prefix in _prepare_qkv to prevent retaining the full prefill allocation.
Generated Inductor code instead returns reinterpret_tensor views of buf18 and
buf47 shaped[1,27,32,128], backed by full[1,16411,32,128] BF16 buffers. buf18 is the
full value projection and buf47 is a separately allocated full key buffer. Across
32blocks those two backing buffers total8,604,090,368bytes(8.013GiB), compared with
14,155,776bytes(13.5MiB) for the intended cloned prefixes. This generated-code
observation strongly explains the measured growth; this run did not directly
instrument every layer cache storage size before cancellation.

Do not promote this compilation mode. A separate candidate could leave extract/
prefill eager and compile only cached steps, or enforce compact cache storage
outside the compiled boundary. Either requires its own bounds, completed output
comparison and second request after park/unpark. Do not raise the VRAM guard or
reduce resolution, precision or step count to hide this failure.

## Host accounting, cleanup and limitations

Compiler RAM was explicitly reserved32GiB, including up to5GiB of disk-cache page
residency. Combined declared123.080GiB fit the benchmark-only128GiB no-swap cgroup.
The ext4 cache was4,476,141bytes(4.269MiB), not tmpfs. Cgroup peak56,097,591,296bytes
(52.245GiB); no cgroup OOM/max event or container OOM kill. CPU maximum71.5°C,
GPU82°C; no thermal flag. The abort was GPU memory pressure, not host RAM or disk.

Cooperative cancellation returned through queue cleanup. Native child/model
allocations were released. At cleanup confirmation Torch allocated8,519,680bytes
but still reserved14,254,342,144bytes; therefore process-local allocator cleanup
alone must not be described as fully releasing physical GPU residency for this
compiled failure. Container exit released that remaining pool. GPU1 returned to
862MiB baseline, and the isolated test containers/Redis/network were then removed.
The compiled candidate is not production-enabled; ordinary component acceptance
and its successful cleanup remain the separately recorded101bc45 result.

Raw evidence, generated compiler code, traces and PNG remain at
`/home/andrew/Documents/Codex/2026-10-08/task-4/image-compile-119-budgeted/`.
The initial96GiB attempt was stopped after eager B before compilation to correct
the compiler accounting; it is preserved separately at image-compile-119/.
