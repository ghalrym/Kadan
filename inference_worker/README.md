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

## Experimental resident image PV TF32

`KADAN_EXPERIMENTAL_IMAGE_PV_TF32` is a default-off, native build experiment.
To opt in, add these options to the existing CMake configuration for
`inference_worker/` and rebuild the worker:

```sh
cmake -S inference_worker -B build/inference-worker \
  -DKADAN_ENABLE_CUDA=ON -DKADAN_ENABLE_IMAGE_GENERATION=ON \
  -DKADAN_EXPERIMENTAL_IMAGE_PV_TF32=ON -DKADAN_ENABLE_ATEN_IMAGE=OFF
```

The image generator passes `image::AttentionPrecision::experimental_pv_tf32`
explicitly to its resident CUDA launchers. Only the PV (probability/value)
SGEMM requests `CUBLAS_TF32_TENSOR_OP_MATH`; the existing scoped restoration
returns the handle to PEDANTIC before the next QK. Operand storage, accumulation,
running numerator and output remain F32, with alpha=1 and beta=1. QK remains
PEDANTIC. Q/K normalization and rotary, softmax, online rescaling, totals,
division, projections and VAE arithmetic are unchanged. Tiles, strides, mixed
text/image masking, full K/V finite scans, status checks, cancellation and
readiness certificates are unchanged. With the option off, both SGEMMs retain
their existing calls and PEDANTIC mode.

Configuration rejects the option without CUDA and image generation, or alongside
ATen image execution, which uses a separate attention backend. If resident
denoising is unavailable or initial admission is rejected, an opted-in request fails
with `image_pv_tf32_requires_resident_attention` before synchronous denoising;
it cannot silently fall back to a different attention policy.

When `KADAN_IMAGE_TIMING=1` is enabled, step records include
`attention_precision_requested`: `pedantic_fp32` or `experimental_pv_tf32`.
This describes the request, including rejected attempts, rather than the
effective kernel. cuBLAS chooses the kernel; `NVIDIA_TF32_OVERRIDE=0` can
suppress TF32 despite the request. Record that override and verify actual PV
kernel selection during separately authorized manual GPU validation.

This experiment has no demonstrated speedup or image-quality acceptance.
PV differences can propagate into later blocks. Default-path exact regression,
opt-in numerical screening, owner visual acceptance and repeated performance
measurements remain separate validation work; unit tests do not establish them.

## Resident image denoising timing

`KADAN_IMAGE_TIMING=1` opts the native image engine into diagnostic timing. It is
off by default and is not an API or UI setting. Disabled stage guards read no
clock, allocate no timing storage on the heap and format no records. Enabled
collection uses fixed per-step storage and steady-clock reads at existing
operator boundaries; it adds no CUDA events, kernels, synchronization or tile
changes.

The worker emits one `KADAN_IMAGE_TIMING ` JSON line per attempted resident step.
The API relays only these bounded lines to its existing stderr/server logs and
keeps its normal diagnostic buffer. Neither progress nor request responses
change. Lines contain request/image/step identifiers, geometry and selected
devices, stage call counts and total/min/max/first/last nanoseconds, step wall
time, unaccounted wall time and deltas of existing transfer-byte counters. Stage
names are `preflight`, `preparation`, `ingress`, `qkv`, `rotary`, `attention`,
`attention_output`, `feed_forward`, `final_output` and `scheduler`. First-step
preflight is included; model loading, prompt encoding, VAE decode and PNG
publication are outside this collector. The host fallback is not profiled.

`completed`, `cancelled`, `failed` and `not_selected` outcomes distinguish full
steps from partial work or rejected resident admission. A stage that throws
includes time spent unwinding its own cleanup. These wall times include host
submission, validation, waits and GPU execution; they are not pure kernel or
DMA timings. Summary formatting/logging occurs after the recorded step wall
interval. Verify the expected complete summary count before using a capture;
diagnostic sink failures cannot replace inference errors, and oversized or
incomplete log lines are dropped. Device lists are capped at 16 and explicitly
mark truncation. A counter reset is flagged rather than reported as unsigned
underflow.

Enabled overhead has not been measured on a GPU. The next separately authorized
measurement should use the existing cached two-step request and unchanged
300-second wall, 88-degree GPU and 120-second API-progress guards. Compare HTTP
and existing completed-block timings with clean #240, report observer/state
variation, and verify explicit unload/cleanup. Do not infer a speedup or a
40-step quality pass from the timing records.

`catalog.json` is the worker’s pinned model metadata. `assets/` contains pinned,
MIT-licensed Whisper tokenizer ranks and mel filters. The C++ conversion reader
accepts only storage/tensor descriptors from verified Whisper archives and does
not execute pickle globals. Curl handles downloads, OpenSSL verifies digests,
and libarchive reads checkpoint containers.

## Muse Spark Open (experimental text support)

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
This entry is an experiment, not a quantization-quality recommendation.

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

An authorized October 15 manual trial on a Threadripper PRO 5955WX with two
24 GiB RTX 3090s verified all ten checkpoint files and loaded the model in
6.6 seconds. Both the original and batched readers reached a 120-second
first-prefill guard without completing one token of the 154-token prompt or
producing output. Cancellation, unload and worker exit completed cleanly.
These trials do not establish usable latency, decode speed, numerical parity
or quantization quality. The CUDA 12.0 trial used the matching host NVIDIA
610.57.04 driver library after an initial mixed-driver initialization failure.

Those trials preceded packed GPU execution. The packed path avoids CPU weight
expansion and expanded F32 host-to-device weight copies for GGUF projections.
It still decodes retained weights into bounded device scratch. A later packed
GPU trial completed 53 of 154 prompt tokens before its 900-second ceiling, with
no output. That trial used token-serial prefill.

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
batched path requires measured prompt completion and output before any latency
or usability claim.
The bounded working set does not
establish interactive usability. Obtain approval before a large download or
manual GPU testing, then use the existing API queue to test
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

All inference is executed by the API-owned worker. Developer generation commands
use `python bin/kadan.py infer` to submit requests through the live API and its
existing queue. There are no standalone model execution tools.
