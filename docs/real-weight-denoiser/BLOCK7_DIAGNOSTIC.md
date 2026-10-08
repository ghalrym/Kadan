# Block 7 FP32 arithmetic isolation

The stopped real-weight gate from commit `9304851657c8e420f117adcccc14b50610ee48bd`
reproduces on one GPU without NCCL. Changing row-wise GEMMs from 128 rows to two
64-row calls causes the observed discrepancy. A diagnostic control that pads each
64-row shard back to the original 128-row geometry is bit-exact at every recorded
stage, including head-sharded attention. This is evidence for shape-dependent
FP32 GEMM arithmetic, not a production fix or a passed real-weight gate.

The saved block-07 capture is SHA-verified against the original manifest. Both
emulated ranks run on physical GPU0, with no transport. Norm/modulation, QKV,
Q/K normalization and RoPE, attention, output projection and residual, and MLP
intermediates are compared against the pinned eager implementation. No whole-block
FP64 oracle is claimed: pinned complex RoPE casts to float32.

## Controls and unchanged tolerance

All comparisons retain `atol=rtol=2e-5`. Each backend uses its own matching eager
reference. Counts below are final-output violations across both emulated ranks.

| Changed operation | Default efficient SDPA | Forced math SDPA |
| --- | ---: | ---: |
| None (manual full path) | 0, bit-exact | 0, bit-exact |
| Normalization partition only | 0, bit-exact | 0, bit-exact |
| QKV GEMMs only | 0 | 0 |
| Attention head partition only | 0, bit-exact | 0, bit-exact |
| Output projection GEMM only | 0 | 0 |
| MLP GEMMs only | 1 | 1 |
| All partitions | **3** | **2** |
| All partitions with original GEMM row geometry | 0, all stages bit-exact | 0, all stages bit-exact |

The last control zero-pads each rank's rows into their original positions, runs
the full-row GEMM, and selects owned rows. It deliberately duplicates work and
is not proposed as a performance optimization. Matching the math attention
backend alone does not eliminate the failure. This experiment excludes NCCL as
the cause of these reproduced coordinates; it does not prove all transport code
correct. Operator profiling does not identify a particular cuBLAS kernel or
reduction algorithm.

## All original failing coordinates

Coordinates use the global 128-row short input; all three are owned by rank 0.
The bound is `2e-5 + 2e-5 * abs(reference)`.

| Coordinate | Actual | Reference | Absolute error | Bound | Error / bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0,39,714] | 0.05514192581176758 | 0.0551152229309082 | 2.6702880859375e-5 | 2.1102304458618167e-5 | **1.2654011751057626** |
| [0,39,2552] | 0.049028992652893066 | 0.04900491237640381 | 2.4080276489257812e-5 | 2.098009824752808e-5 | 1.1477675750205318 |
| [0,39,3323] | 0.31084251403808594 | 0.31081438064575195 | 2.8133392333984375e-5 | 2.621628761291504e-5 | 1.0731264757762615 |

The largest absolute error remains 0.0001983642578125 at a different coordinate
(reference 229.69900512695312, actual 229.6988067626953). That point passes its
relative bound. The JSON records both the maximum normalized coordinate and all
violations. Forced math has two violations, [0,39,2552] and [0,39,3013], and a
maximum normalized ratio of 1.1420860115771156; it is a separate diagnostic.

## Where the drift grows

Default all-partition maximum absolute errors start at zero for norm1/mod1,
then Q/K/V projections reach 1.1444e-5 / 1.3351e-5 / 4.7684e-6.
Attention reaches 2.2888e-5, output projection/residual1 1.0681e-4,
MLP product 7.2479e-5, and MLP output 2.2888e-4.
Replacing the split path's MLP product or MLP output with its eager counterpart
removes final violations, while retaining other nonzero differences.

An isolated CPU FP64 dot uses each path's exact saved FP32 MLP-product operands
and captured weight row, promoted to FP64. At [0,39,714], both residuals are
-2.483762741088867 and the saved GPU gate is -0.9962648749351501. Eager MLP output
is -2.548396587371826 versus its FP64 dot -2.5483958389050754; split MLP output
is -2.5484232902526855 versus its FP64 dot -2.548414451209397. Thus upstream
operand drift and GEMM rounding both contribute before cancellation with the
residual produces the small final output. This checks only the isolated final
MLP dot, not the accuracy of the entire block. All failing-point dot traces are
retained in `evidence-block7/coordinate-trace-fp64.json`.

## Settings and execution evidence

Pinned image `sha256:eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa`;
PyTorch 2.14.1+cu130, CUDA 13.0, RTX 3090. Matmul TF32 is disabled and float32
matmul precision is highest. cuDNN TF32 remains enabled, but the observed default
SDPA operator is `_scaled_dot_product_efficient_attention`; the control uses
`_scaled_dot_product_attention_math`. FP16/BF16 GEMM reduced-precision flags are
enabled; math-SDPA reduced precision is disabled. CUBLAS_WORKSPACE_CONFIG is unset.
Exact settings and observed operators are in the JSON evidence.

The complete GPU0 control run was 2026-10-08 22:14:24.9906–22:14:48.1373 UTC,
container exit 0; diagnostic work took 6.384 seconds with 1,147,815,424 bytes peak
allocated and 1,172,307,968 bytes peak reserved. The host launcher bounded execution
to 150 seconds, 8 GiB host RAM/no swap, and four CPUs; the probe bounded its CUDA
allocator to 4 GiB. It exposed only GPU0 and removed its container afterward.
The API was never paused: exact container ID
`1efcf5729f776dd90bbc0611c63a474bc696cadc0d48b9011fbb320e167b6499`, StartedAt
`2026-10-08T22:03:44.290735994Z`, remained healthy on GPU1. Evidence includes resource
samples and the preserved API identity. Large tensor captures remain local.

`validate_controls` retains the three default failures and two math failures,
while requiring the shape control to be bit-exact at every recorded stage. The
post-run assertions were validated against saved reports on CPU. Unit tests cover
row ownership/geometry and the distinction between absolute and normalized worst
coordinates. The first CPU test invocation mounted the harness too shallowly for
an existing launch-path test; rerunning with the full repository mount corrects
that test environment without source changes. The repository is also set as the
container working directory to avoid importing the image’s older `/app/api`.

No production code, tolerance, reference gate, BF16/full-chain/full-image status,
or live deployment changed. Review an arithmetic/reference strategy before
advancing the real-weight ladder. `diagnostic_launcher.py` is a host-specific
reproduction helper with an explicit execution flag and fresh-output requirement.
