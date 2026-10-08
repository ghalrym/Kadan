# Synthetic GPU residency verification

Executed 2026-10-08 on an RTX 3090 (SM86), CUDA 12.0.140, GCC 12.4.0,
Debug build. Only the new `kadan-resident-parity` target and its native dependencies
were built. This explicit GPU executable is not in automatic CTest: it requires
an operator-selected, available device and rejects anything except a tiny
four-layer/16-hidden/16-vocabulary/four-expert checkpoint.

Reproduction (choose an available GPU UUID; do not target a live worker):

```sh
python3 native/tests/resident_parity_fixture.py /tmp/kadan-new-parity-fixture
cmake -S native -B /tmp/kadan-resident-cuda -DKADAN_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=OFF \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build /tmp/kadan-resident-cuda --target kadan-resident-parity --parallel 2
CUDA_VISIBLE_DEVICES=SELECTED_GPU_UUID timeout 75s \
  /tmp/kadan-resident-cuda/kadan-resident-parity --execute \
  /tmp/kadan-new-parity-fixture/good /tmp/kadan-new-parity-fixture/bad
```

The checked run used GPU1, UUID `GPU-2a2378dd-08c1-6f69-6317-a253d90e76b3`.
GPU0's live `/opt/kadan/local/kadan-model-worker`, PID 1452475, remained at
21,508 MiB; GPU0 total usage remained 21,779 MiB before/after. The synthetic
process exited and disappeared from the process list. GPU1 desktop usage changed
from 851 to 857 MiB, so whole-device readings are not exclusively ours.

Exit status 0, elapsed 0.706101 seconds:

```text
PARK model_gpu_allocations=0 context_envelope=536870912 actual_gpu_free=24112463872
PASS split-vs-legacy: 4 cycles x 3 tokens x 16 logits bit-exact; selections/EOS/progress equal; arena=54784 retained_ram=14668 weight_device_with_headroom=536913920 gpu_free_baseline=24112463872 gpu_free_final=24114561024; cancelled-step recovery, partial-load failure and budget rejection cleaned; zero final reservations
```

All 192 logits are bit-identical to the legacy execution, including after request
state recreation, park/reload and cancellation recovery. The fixture also checks
12 token selections, EOS flags and committed progress. It verifies real-device
cleanup after a NaN projection-scale load failure and failed budget admission.
Driver synchronization/free/reset failures are injected only in CPU tests.
CPU tests separately compare every serialized dense/FP8/NVFP4 weight and scale
region, pointer role and NVFP4 global multiplier across lifecycle transitions;
fake numerical kernels alone would not detect misbound weights.

At park, model allocations are zero but a 512 MiB context/headroom envelope stays
charged. Only successful dedicated-context teardown releases it. Driver free-memory
queries after reset may initialize a diagnostic context; ledger zero is not a
claim that a running process has literally zero total driver/context residency.
The harness exits afterward. CPU tests verify another owner cannot spend the
parked envelope and failed context reset keeps it reserved and quarantined.

SHA256 evidence for the checked run:

| Artifact | SHA256 |
| --- | --- |
| GPU executable | `c2e911250faecff8d5680ab257936edec88c3543bb2fc1c9a16bcbccfc526db8` |
| Valid synthetic safetensors (56,805 bytes) | `0d266c63a5f0ae29d63b51642a0aaf1f58055a1809879741dcb11591ee2c621c` |
| Invalid-scale safetensors | `b6d066e5d3bcc4db93a30b3b5369000602ca10e1e9a856115c373d091be05d9c` |

This is correctness evidence for tiny synthetic weights, not a production-model
benchmark or text quality measurement. The live 51.92 decode tokens/sec is prior
legacy evidence. The installed GCC12 Release build encountered an existing
`std::sort` array-bounds warning in `cuda_moe.cpp`; Debug was used without changing
those kernels or suppressing their warnings. No model download, production
restart, image rebuild or database write was performed.
