# Original SM86 NVFP4 fused projection

This milestone implements Kadan's first original fused quantized GPU projection:
one CUDA kernel reads canonical ModelOpt NVFP4 weights/scales, dequantizes each
pair in registers, multiplies by an FP32 input vector and reduces one output row.
It uses CUDA cores on SM86, not hardware FP4 tensor cores. No external inference
implementation or kernel is copied, wrapped, vendored or ported.

The worker/API is not switched to this path. The optional C++ wrapper is an
explicit GPU-executing API, and the parity executable is deliberately absent from
CTest. CPU correctness and successful compilation do not establish GPU parity,
performance or suitability for the full Qwen model.

## Implementation

`nvfp4_kernel.cu` launches 128 threads per row. Each lane reads strided packed-byte
pairs and their shared 16-column E4M3FN scale, computes the two E2M1 weights in
registers, rounds the global-scale multiplication to FP32 and accumulates with
FP32 FMA. Four warp reductions feed four shared floats; a final warp reduction
writes the row result. All lanes participate, including on short/tail rows.
There are no host projection tiles or dense dequantization buffers. A four-byte
status flag reports decoded weights whose exact pre-round value exceeds FP32
range, or nonfinite reductions. The range check uses an exact double product
before FP32 rounding, matching the CPU decoded-weight contract.

The supported layout is exactly #99/#100's canonical unswizzled ModelOpt NVFP4
2D view with one positive finite global multiplier. Input and output are FP32.
A separate [FP8 projection milestone](FP8.md) now shares the resource owner.
BF16 output/activation rounding, activation quantization, bias,
batching, expert routing, graph capture, tensor cores and model integration are
not implemented. Host validation rejects unsupported layouts and bad scales.
GPU accumulation is FP32 rather than the CPU reference's double sum, so reduction
rounding differs and FP32 accumulation can overflow even when a double sum would
not. Reported numerical failure is an error, not a fabricated result.

## Memory, ownership and error boundaries

`plan_nvfp4` is a CPU-only function. It checks the canonical view and grid limits,
then computes overflow-checked, 256-byte-aligned regions for packed weights,
block scales, input, output and status in one allocation. Its explicit byte cap
includes padding. `Nvfp4Projection` reserves exactly that allocation from the
supplied #98 `Resources` ledger before CUDA allocation/runtime validation. Only
the selected device's budget is charged; RAM and other GPU budgets are untouched.
The caller must have already admitted its host buffers and retain context/driver
headroom. This is requested-allocation accounting, not measured VRAM usage.

The creating thread must select the intended device before construction. The
wrapper verifies that current device and requires compute capability 8.6. It
never changes device selection, resets a device, changes power settings, or owns
the primary context. Methods and destruction belong to that same thread/device.
The caller retains the shared global resource manager independently of the
projection, so quarantined reservations remain visible after object destruction.

Construction uploads the two immutable host weight/scale spans and synchronizes
before returning; no host weight pointers are retained. The global multiplier is
copied by value. One persistent device slab also holds input/output/status. Each
matvec pins the resident, copies the input, launches once, synchronizes, checks
the numerical status and copies the result to caller-owned RAM. No per-call
allocation is made. Input/output may alias; output is unspecified on error and
must be discarded. No host span is referenced by the kernel.

This first implementation uses explicit legacy-default-stream kernel/memset/
synchronization and the corresponding blocking `cudaMemcpy` API. Per-thread
default-stream API overrides are rejected at compilation. It is deliberately a
blocking boundary, not a concurrent-stream or CUDA-graph scheduler. Default-stream
serialization may wait for other work in the same context; production scheduling
and private-stream ownership remain separate work.

Launch/runtime errors poison the object and retain its active pin until cleanup
can synchronize successfully. `close()` synchronizes, removes the pin, begins
eviction, frees the allocation, and only then releases the reservation. If
synchronization/free/ownership checks fail, cleanup is latched as uncertain: the
reservation is retained, no pointer retry/reset is attempted, and explicit close
reports failure. The noexcept destructor makes the same best-effort cleanup but
cannot report an exception. Callers must explicitly close to observe failures;
a supervisor must treat uncertain cleanup as fatal worker state rather than
reusing that memory budget. Fault-injection/real CUDA error paths remain untested.

## Build and verified toolchain

Installed tools inspected without device probing: CUDA 12.0, nvcc 12.0.140,
CUDART 12.0.146, GCC 12 (NVCC host compiler), GCC 13.3 (ordinary C++), CMake 3.28.3.
The kernel, owning wrapper and parity executable compile and link for SM86 with
these tools. No tool installation was needed. Revision `9e4161d` later passed
the six explicitly authorized tiny NVFP4 GPU cases below. That approval is
consumed; it does not validate or authorize the subsequent FP8/shared-owner
revision. See [FP8.md](FP8.md) for the proposed combined validation batch.

CPU-only tests remain the default and require no CUDA toolkit:

```sh
cmake -S native -B /tmp/kadan-native-cpu -DKADAN_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/kadan-native-cpu --parallel 2
ctest --test-dir /tmp/kadan-native-cpu --output-on-failure
```

Compile-only CUDA validation (does not run the resulting executable):

```sh
cmake -S native -B /tmp/kadan-native-cuda -DKADAN_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/kadan-native-cuda --parallel 2
```

CUDA builds require explicit/fixed SM86 targeting; `native` architecture detection
is rejected. The build preserves subnormals (`--ftz=false`) and uses legacy stream
semantics. CI runs the CPU-only suite including allocation-plan limits; it does
not install a CUDA toolkit or execute GPU tests. CUDA compilation was local.

## Prepared, unexecuted GPU validation

**Do not run until Andrew explicitly authorizes GPU validation after the breaker
trip.** The smallest proposed validation is one invocation on one approved RTX
3090, six synthetic kernel launches total, no model files, no loops for timing,
no benchmarks and no simultaneous second GPU. The largest case is 33 × 2048;
operator allocations are capped at 64 KiB and fixture RAM remains below 1 MiB.
CUDA context/driver/module overhead is additional and has not been measured.

After authorization only:

```sh
/tmp/kadan-native-cuda/kadan-cuda-parity --allow-gpu-validation --device 0
```

The executable requires the opt-in flag. Four deterministic shapes exercise
short rows, multiple rows, all FP4 codes, block-scale extremes, and a non-power-of-
256 tail: `(3,32)`, `(5,256)`, `(33,2048)`, `(2,2064)`. The global scale
is non-power-of-two. Results are compared with #99's CPU reference using
`1e-5 + 64 * FLT_EPSILON * sum(abs(weight * input))` per row, bounding reduction
error even near cancellation. A fifth `(1,16)` fixture checks the exact subnormal
result with no tolerance, detecting accidental flush-to-zero. The sixth `(2,16)`
case expects overflow for positive and negative decoded weights that exceed
FP32 range before rounding yet round to finite FLT_MAX in magnitude, using zero
input. It replaces the ordinary `(1,16)` case, keeping exactly six launches.
Each projection
closes and verifies that its reservation was released. This harness has been
executed once successfully at `9e4161d`, including normal reservation cleanup.
The later shared-owner revision has only CPU/build validation. Runtime-failure
cleanup remains unverified. Do not interpret a later parity pass as a throughput benchmark.

CUDA API/warp semantics reference: [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).
Weight-format references and the CPU numerical contract are in
[QUANTIZATION.md](QUANTIZATION.md). Full Qwen generation and the dual-3090 50+
decode tok/sec target remain unmeasured.
