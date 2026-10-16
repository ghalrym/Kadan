# Inference worker

The API uses one `kadan-inference-worker` process for chat, Decisions, image,
video, transcription and speech. It owns the FIFO, checkpoint acquisition,
input preparation, engine lifecycle, result packaging and workspace cleanup.
The API submits serialized Pydantic tasks and relays events and results.

## Source layout

`src/inference_worker.cpp` is the process entry point. Implementation files are
grouped by responsibility:

- `src/requests/`: request parsing, preparation and response formatting.
- `src/inference/`: per-modality engine orchestration.
- `src/preparation/`: checkpoint downloads, asset preparation and input loading.
- `src/weights/`: checkpoint storage, manifests, metadata and quantization.
- `src/models/`: model implementations in `text/`, `image/`, `video/`, `speech/`
  and `decision/` subfolders.
- `src/tokenization/`: shared text tokenization.
- `src/media/`: shared media sessions and residency management.
- `src/cuda/`: GPU implementations, kernels and their private headers.

Private headers live beside their implementations. Public headers remain under
`include/kadan/`; CMake lists source files explicitly in the existing targets.

## Request handling

`src/requests/inference_request.cpp` decodes the transport envelope into named C++ requests;
`include/kadan/inference_owner.hpp` processes the queue head. Each modality's
`src/requests/inference_*_request.cpp` owns its preparation and response formatting alongside
its engine. Chat messages, transcription PCM and typed decision results pass directly
in memory. Decision answers are formatted as JSON once for the response.
`JobWorkspace` owns only workspace paths and cleanup.

`src/preparation/model_store.cpp` is the single checkpoint downloader for both inference and
Settings download requests. It also owns checkpoint validation, context limits and
model selection. The worker publishes status snapshots at startup and after tasks;
the API serves those snapshots, observes download progress and forwards cancellation.
Settings changes use typed queue tasks.
`src/preparation/whisper_export.cpp` and `src/preparation/tokenizer_assets.cpp` convert checkpoint metadata
in C++. The worker never starts a Python interpreter.
Cancellation finishes owned work and releases the workspace before the next job.

## Image execution

Image generation uses C++ ATen in the existing worker. The complete worker build
requires LibTorch 2.14.1+cu130 and CUDA 13.0; `api/Dockerfile` supplies these by
default. `KADAN_ENABLE_IMAGE_GENERATION=ON` requires CUDA and the pinned Torch
package. CPU unit and metadata-tool builds leave generation disabled. Missing
dependencies fail configuration; there is no hidden CPU image fallback.

The accepted October 13 Turbo fox comparison used ATen with TF32 disabled.
See the [image execution documentation](../README.md#c-aten-image-execution) for
its exact build, workload and separate generation/load/unload measurements.
ATen keeps TF32 disabled for FP32 matrix multiplication and cuDNN operations,
and keeps BF16 reduced-precision reduction disabled. The former image PV TF32 option and custom image generation backend
have been removed. Shared media sessions, projection kernels and decoder-boundary
utilities remain for their other callers and component unit tests.

`KADAN_ATEN_IMAGE_TIMING` stderr records report synchronized component loading,
generation and unloading. The API relays bounded timing lines to its logs.
Resident timing utilities and their unit checks remain available for component
analysis, but `KADAN_IMAGE_TIMING` no longer selects timing in the image worker.

`catalog.json` is the worker’s pinned model metadata. `assets/` contains pinned,
MIT-licensed Whisper tokenizer ranks and mel filters. The C++ conversion reader
accepts only storage/tensor descriptors from verified Whisper archives and does
not execute pickle globals. Curl handles downloads, OpenSSL verifies digests,
and libarchive reads checkpoint containers.

## Muse Spark Open text support

The native Muse engine uses the existing worker, resource ledger and bounded
CUDA projection banks. GGUF projections read bounded packed rows, retain their
compressed bytes in the existing GPU banks and decode bounded F32 tiles on the
GPU for the existing GEMM path. F32 tensors and MLX use their existing sources. It does not start llama.cpp, MLX or a Python runtime.
The current model selection, saved context limits and GPU settings remain
unchanged.

The optional `muse-spark-open-q3-xl` catalog entry pins
[Unsloth UD-Q3_K_XL](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF/tree/19d0b1b3a1b1974a48439e11e0639c0dad164fc8/UD-Q3_K_XL).
Its ten files total 401,483,491,832 bytes (373.911 GiB). Its published tensor
inventory contains 1,429 tensors: expert gate/up matrices use Q3_K, expert down
matrices use Q4_K, other matrices use Q8_0 and vectors use F32. This is a mixed
quantization preset; the loader rejects unsupported wire types or geometry.
The measurements below do not establish quantization quality.
The existing Settings and model-selection API accept this catalog ID through
the shared `LlmModel` type; selection uses the existing worker queue.

Acquisition downloads the tensor-free first split before any weight split,
validates its architecture and exports configuration and byte-level BPE assets
in C++. The tokenizer uses that split's own vocabulary, merges and control
IDs. The native prompt renderer matches the complete pinned public chat
template, and the recipient parser separates reasoning, commentary and final
text while suppressing unsupported tool messages. The existing worker's
2,048-token extra reasoning budget is independent of the template's default
reasoning strength of 512. Generation remains greedy.

`src/models/text/muse_spark.cpp` implements the 62-layer text decoder and
handles the GGUF converter's absolute norm gains, folded router scale,
precomputed residual coefficients and Q/K row ordering explicitly. GGUF uses
F32 activation arithmetic. The alternate public MLX affine group-64 3/4-bit
layout is also supported natively, with its BF16 activation boundaries; it is
not registered as a selected model. Original BF16, ModelOpt NVFP4, other GGUF
presets and multimodal execution are not admitted by this implementation.

The pinned `ggml-base` dependency supplies GGUF metadata parsing and quantized
CPU row decoding only; original Q3_K, Q4_K and Q8_0 CUDA decoders serve packed
projections. CPU unit tests compare the shared decoder arithmetic against the
pinned reference, including FP16 patterns and randomized blocks. Kadan owns
model execution, attention state, placement,
cancellation and sampling. Its MIT notice is installed with the worker.
Reference equations and conversion semantics were checked against
[Transformers PR 49644](https://github.com/huggingface/transformers/pull/49644),
[MLX-LM PR 1975](https://github.com/ml-explore/mlx-lm/pull/1975) and
[llama.cpp PR 30616](https://github.com/ggml-org/llama.cpp/pull/30616).

Metadata-only unit fixtures cover the published MLX and GGUF inventories using
sparse files with unread payload holes. Separate units check quantized block
decoding, tokenizer metadata, prompt rendering, recipient streams and
pre-download rejection. No unit test executes the model.

Muse now prefills bounded prompt chunks layer by layer, reusing each projection
across token rows. Chunks contain at most 256 tokens within the existing 128 MiB
host scratch envelope. Attention writes and consumes K/V in absolute-position
order, and expert gathering retains each token's ascending-expert accumulation.
Scalar decode shares the same layer math. Only the final prompt row computes
vocabulary logits. Component units compare causal rings, routing and scatter
against independent serial calculations; batched GEMM can change floating-point
reductions, so these checks do not establish whole-model numerical parity.

Native `prefill_layer` progress reports completed layers across chunks;
`prefill_chunk` and `prefill_token` report committed tokens only. Muse
`decode_token` progress counts computed predictions even when recipient parsing
suppresses control or reasoning tokens. These events support a bounded manual
stall guard without describing partially computed tokens as complete. The
batched path has the measured limits below.

An authorized October 16 manual retest used the same native build tree as
`1ba4dd16a59e71a924fd25f0412c48d710fdbb7a`, the pinned ten-file checkpoint,
512-token context, greedy sampling and the prior 154-token greeting prompt on a
Threadripper PRO 5955WX with two 24 GiB RTX 3090s and 448 GiB recognized RAM.
The system prompt was `Reasoning strength: 0. Answer directly.` and the user
prompt was `Reply with exactly: Hello Andrew.`; these were retained for
comparison, not as a recommended reasoning configuration.

| Stage | Measured result |
| --- | --- |
| Model load/setup | 1.62 seconds |
| Prefill, excluding load and setup | 100.96 seconds for 154 tokens |
| Prompt completion from worker launch | 102.87 seconds |
| Computed predictions | 62, including recipient control and reasoning tokens |
| Warm decode | 0.330 predictions/second over warm intervals; median 1.507 seconds |
| Useful complete answer within five minutes | No; coherent but unfinished reasoning, no final answer |
| Cleanup | Cancel at 290 seconds; unload acknowledged zero memory usage and clean exit at 292.19 seconds |

Warm intervals exclude the first decode interval between predictions one and
two. The initial faster decode rate did not persist: the last ten intervals
had a median of 5.64 seconds. Load/setup does not preload all weights; roughly
232 GB of cumulative storage reads were recorded near prompt completion, so
prefill includes cold checkpoint I/O. The retest followed a reboot and removal
of a DIMM from a known faulty slot. Cache residency and automatic GPU placement
differed; the result does not isolate the cause of the improvement or establish
whole-model numerical parity, quantization quality or interactive usability.

The best tested packed GPU path is on `master`. No live deployment is implied by
the native implementation or this manual measurement.
Obtain approval before a large download or manual GPU testing, then use the
existing API queue to test
load, short and streamed chat, history, reasoning, cancellation, unload and
context admission. Record actual memory and loading/prefill/decode times.

## GPT-OSS

The catalog includes the official OpenAI Transformers-layout checkpoints below.
The existing `medium` ID continues to identify 120B so its cached checkpoint and
saved settings retain their identity.

| Catalog ID | Checkpoint | Published total / active parameters | Download bytes |
| --- | --- | --- | --- |
| `gpt-oss-20b` | [OpenAI GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee) | 21B / 3.6B | 13,789,263,104 |
| `medium` | [OpenAI GPT-OSS 120B](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a) | 117B / 5.1B | 65,276,857,840 |

Both are Apache-2.0 licensed; each catalog entry links the license at its pinned
revision. Byte counts sum the Hub's file sizes for root files accepted by
`ModelStore::allowed`, including weights, tokenizer, configuration and license
files. They exclude `.gitattributes` and the separate `original/` checkpoint.
BF16 dequantization, GGUF and third-party requantizations are not registered
choices; choosing a different quantization requires a separate loader/quality
discussion.

`src/cuda/gpt_oss_model.cpp` executes both sizes (`gpt_oss` /
`GptOssForCausalLM`) in the existing chat engine with batch-one CUDA kernels in
`src/cuda/gpt_oss_kernel.cu`: RMSNorm, fused BF16 QKV with biases, half-split
YaRN rotary, grouped-query attention with per-head sinks over a BF16 K/V cache
(128-slot ring on sliding layers), and top-4 routed experts that read the
checkpoint's packed MXFP4 blocks directly with the clamped SwiGLU. Activations
and accumulation are FP32; selection is greedy.

`src/models/text/gpt_oss_plan.cpp` places complete layers in contiguous ranges
that end, with the LM head, on the largest device. It prefers one device, then
several, and otherwise keeps each layer's attention and context state on the
GPU while experts that do not fit stay in mapped pinned host memory, which the
same kernels read across the bus. The embedding table is always host-mapped.
These host bytes are admitted before loading.

`include/kadan/harmony.hpp` renders prompts like OpenAI's `openai-harmony`
encoder (system messages become developer instructions; earlier reasoning is
dropped) with low reasoning effort, and converts generated channels to text:
analysis streams inside `<think>…</think>`, final and preamble commentary stay
visible, and tool-call bodies are never emitted. Reasoning has a 2,048-token
budget beside the normal answer limit. `<|return|>`, `<|call|>` and
`<|endoftext|>` stop generation. Tools are not part of the chat request model.

No Kadan speed or memory-fit figure has been measured. Manually test each size
through the existing API queue: loading and automatic placement (including host
experts for 120B), basic and streamed chat, multi-turn history, reasoning
separation, cancellation, unload and context limits, recording actual memory and
latency.

Build and install the `inference-runtime` component using the options in
[`api/Dockerfile`](../api/Dockerfile). The former standalone worker executables
and launchers have been removed. Engine libraries remain here; metadata inspection tool
sources live in `../bin/cpp/` and are exposed through `../bin/kadan.py native`.
Run `python bin/kadan.py native build` from the repository root to compile them.
Normal runtime builds leave `KADAN_BUILD_TOOLS` disabled.

### Weight-cache diagnostics

Developer builds can enable `KADAN_TRACE_WEIGHT_CACHE` to emit ordered cache
records on worker stderr. It defaults off and changes no runtime cache policy.
Capture the complete process stderr from bank creation through its final cleanup
summary, then run `python bin/kadan.py native cache-replay TRACE` on the raw JSON
lines. Existing controller envelopes must first be reduced to their `line` values.
Trace output is limited to 131,072 ordinary records, 1,024 summaries and two
truncation notices per GPU. The replay rejects truncated or incomplete captures.

Replay compares first admission, byte-bounded LRU and a documented experimental
decayed tile-frequency filter at the same bank capacity. Its transfer counter is
ideal missed-weight demand, not measured PCIe traffic: packed GGUF admission may
upload the same bytes again. Include admission traffic and actual copy timings
before choosing a runtime policy. The replay does not model in-flight pointer
protection or preserving residency across existing retirement boundaries, and
does not cover the separate GPT-OSS or AutoModel placement paths.

All inference is executed by the API-owned worker. Developer generation commands
use `python bin/kadan.py infer` to submit requests through the live API and its
existing queue. There are no standalone model execution tools.
