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

## GPT-OSS checkpoint choices

The catalog includes the official OpenAI Transformers-layout checkpoints below.
Both are **download only** until the native worker supports GPT-OSS. Adding these
entries does not select a model or download weights. The existing `medium` ID
continues to identify 120B so its cached checkpoint and saved settings retain
their identity.

| Catalog ID | Checkpoint | Published total / active parameters | Download bytes |
| --- | --- | --- | --- |
| `gpt-oss-20b` | [OpenAI GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee) | 21B / 3.6B | 13,789,263,104 |
| `medium` | [OpenAI GPT-OSS 120B](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a) | 117B / 5.1B | 65,276,857,840 |

Both are Apache-2.0 licensed; each catalog entry links the license at its pinned
revision. Byte counts sum the Hub's file sizes for root files accepted by
`ModelStore::allowed`, including weights, tokenizer, configuration and license
files. They exclude `.gitattributes` and the separate `original/` checkpoint.
They describe downloads, not resident memory, KV cache, workspace or dual-3090
fit. No Kadan speed or memory-fit claim has been measured for either model.

The pinned [20B configuration](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json)
and [120B configuration](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json)
declare `gpt_oss` / `GptOssForCausalLM` with MXFP4 expert weights. This is a
mixed-precision checkpoint, not every tensor stored at four bits, and MXFP4 is
not NVFP4. These are the only GPT-OSS weight variants offered here. BF16
dequantization, GGUF and third-party requantizations are not registered choices;
choosing a different quantization requires a separate loader/quality discussion.

Kadan's root downloader accepts these zero-indexed safetensor shards, but its
current Qwen/GLM engines do not execute them. Enabling inference requires native
GPT-OSS architecture and MXFP4 support, its tokenizer and
[Harmony prompt/output protocol](https://github.com/openai/harmony), reasoning
channel and tool-turn handling, and explicit API model-ID support for 20B.
Keep `inference_available` false until those capabilities are implemented.
Execution must remain in the existing single worker with automatic placement.

After runtime support and an owner-approved weight choice, manually test each
size through the existing API queue: loading and automatic placement across the
two 3090s (including memory admission/offload), basic and streamed chat, multi-turn
history, reasoning levels/channel separation, tool-call round trips, cancellation
and unload. Test context limits and record actual memory and latency. Neither
generation nor GPU testing is part of this catalog change.

Build and install the `inference-runtime` component using the options in
[`api/Dockerfile`](../api/Dockerfile). The former standalone worker executables
and launchers have been removed. Engine libraries remain here; metadata inspection tool
sources live in `../bin/cpp/` and are exposed through `../bin/kadan.py native`.
Run `python bin/kadan.py native build` from the repository root to compile them.
Normal runtime builds leave `KADAN_BUILD_TOOLS` disabled.

All inference is executed by the API-owned worker. Developer generation commands
use `python bin/kadan.py infer` to submit requests through the live API and its
existing queue. There are no standalone model execution tools.
