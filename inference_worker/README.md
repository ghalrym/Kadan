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
Lower-memory GPUs stream operator weights and tile the VAE convolutional stages
when whole-component residency or full-resolution workspace does not fit. All
VAE arithmetic and global bottleneck attention stay on the GPU. The shared ledger
admits host transfers, weight residency and tiled workspace before execution.

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

## Hardware tier text models

Settings offers four hardware tiers. Each tier in `catalog.json` pins one GGUF
language model, Qwen-Image-2.1-Turbo (the only image model), H3 FL2VA INT8 + Turbo,
the Qwen3-TTS 1.7B CustomVoice speech model, Whisper medium for transcription
(also the default without a saved Whisper choice) and the S1-mini formatter.
Choosing a tier persists `tier.json` in the model store and selects its language
and image models; the API then queues the tier's incomplete downloads through
the existing FIFO.

| Tier | Hardware | Catalog ID | Checkpoint | Download bytes |
| --- | --- | --- | --- | --- |
| Go | 8 GB VRAM, 32 GB RAM | `qwen3.8-27b-q4-k-m` | [ggml-org Qwen3.8-27B Q4_K_M](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/71bc7b627595dc8a91039addd9c791ae548d6747) | 18,973,870,528 |
| Plus | 16 GB VRAM, 64 GB RAM | `qwen3.8-27b-q8-0` | [ggml-org Qwen3.8-27B Q8_0](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/71bc7b627595dc8a91039addd9c791ae548d6747) | 28,595,763,648 |
| Pro | 24 GB VRAM, 128 GB RAM | `qwen3.8-flash-next-q3-k-xl` | [Unsloth Qwen3.8 Flash Next UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q3_K_XL) | 89,986,353,824 |
| Ultra | 48 GB VRAM, 512 GB RAM | `glm-5.3-flash-q3-k-xl` | [Unsloth GLM-5.3 Flash UD-Q3_K_XL](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/a38483c8cd5df544f53d70fb281afe97369d5ab6/UD-Q3_K_XL) | 147,535,922,080 |

Qwen3.8 Flash Next is under the Qwen Community License 1.0; its catalog notice
must be acknowledged before download.

Acquisition downloads the first GGUF file (an unsplit file, or a tensor-free
first split) before any weight split. `src/preparation/gguf_assets.cpp`
validates the exact architecture contract and derives `config.json` and a
byte-level BPE `tokenizer.json` from the GGUF tokenizer arrays: normal tokens
form the vocabulary, control and user-defined tokens are added tokens, and the
published pre-tokenizers (Qwen NFC with `\p{M}`, GLM with `ignore_merges`) are
reproduced. The derivation matches the upstream `tokenizer.json` vocabularies,
merges and added tokens exactly.

`src/models/text/` holds one engine (`gguf_engine.cpp`) and one model per GGUF
architecture, following llama.cpp
`6420cf5dd588c68e48657a0bbd82bb3bd5ac7209` graphs and converter conventions:

- `qwen35.cpp`: Qwen3.8 27B, 48 gated delta net layers and 16 gated full
  attention layers with a dense SwiGLU MLP.
- `qwen4exp.cpp`: Qwen3.8 Flash Next, four hyper-connection streams in place of
  layer norms, n-gram hash embeddings on layer 1, a 512-expert top-10 MoE with a
  gated shared expert, and QSA top-block selection on full attention layers.
- `glm.cpp`: GLM-5.3 Flash, mHC streams, KDA linear layers and nope-only MLA
  with the k-pool sparse indexer. MLA is absorbed, so only the latent is cached.

`qwen_layers.cpp` holds the delta net and gated attention shared by both Qwen
models. Layer math is F32 on the host; text-only interleaved M-RoPE reduces to
NEOX rotation of the first 64 dimensions. Projections run through the existing
device banks, which keep packed rows and decode F32, Q8_0, Q3_K, Q4_K, Q6_K,
IQ3_XXS, IQ4_NL, IQ4_XS and BF16 on the GPU. Attention uses the device attention
path when available.

Checkpoints larger than VRAM are tiered across VRAM, host RAM and disk by
`gguf_placement.cpp`. Each slice (one expert of an expert stack, or one matrix)
carries a usage score that gains one per projection and halves every 512
tokens. A slice stays device-resident while it fits 90% of the planned bank
capacity; a resident slice is released only for one scoring 1.5x higher, so
routinely routed experts stay in VRAM and rare ones do not thrash it. Slices
without device residency are held in a host RAM cache with the same admission
rule and stream to the GPU without displacing resident tiles; anything else is
read from the GGUF files. The chat engine admits host memory left after the
model's state and a 4 GiB headroom (capped at the checkpoint size) as that RAM
cache in the shared resource ledger.
Prefill is layer-major in chunks of 64 tokens; selection is greedy. MTP heads,
vision towers and Hadamard-folded checkpoints are not executed.

Unit tests cover the device decoders bit for bit against ggml, checkpoint
binding, metadata validation, asset derivation and the shared layer math. No
model has been executed: whole-model numerical parity, quantization quality,
speed and memory fit on each tier are unmeasured. Manually test each tier
through the existing API queue: download, load, short and streamed chat,
multi-turn history, cancellation, unload and context admission, recording
actual memory and loading, prefill and decode times.

## Hardware tier video

All four tiers use the existing pinned `h3-fl2va-int8-turbo` audiovisual bundle.
The catalog owns the clip defaults exposed by Settings and used by the video
page and requests that omit duration or resolution:

| Tier | VRAM / RAM | Default clip |
| --- | --- | --- |
| Go | 8 GB / 32 GB | 480p, 5 seconds |
| Plus | 16 GB / 64 GB | 480p, 8 seconds |
| Pro | 24 GB / 128 GB | 768p, 5 seconds |
| Ultra | 48 GB / 512 GB | 768p, 8 seconds |

These are starting settings. Every tier accepts explicit 1–15 second, 480p or
768p requests; admission uses the actual request geometry, prompt and available
memory. A larger request can wait for resources rather than fit its nominal tier.
The native worker retains the existing queue, cancellation, joint audio/video
denoising, stereo audio decoding and MP4 packaging.

The supported H3 execution format is ConvRot INT8 for text and denoiser
projections, with the existing BF16/F32 dense weights and Turbo adapter, FP16
video VAE and F32 audio VAE. Checkpoints stay on disk and projections stream in
bounded slices. GPU workspaces and weight banks are admitted per physical
device; two 24 GB devices do not provide one contiguous 48 GB allocation.
Text encoding, denoising, audio decoding and temporally chunked video decoding
run in sequence. Existing resident denoiser stages use device memory when their
admission succeeds. The F32/F64 precision policy is unchanged.

The host cache uses half the memory remaining after the request's execution
envelope, capped by the serialized and decoded weight inventory. When the full
inventory does not fit, completed stages retire their cached tensors so the
next stage can reuse the cache. Host and device retention are decided separately.
Larger machines retain completed stages when the full inventory fits. Device
retirement preserves bank partitions and reservations; it makes room for later
weights, without enlarging the resident activation budget. Each request checks
physical and ledger headroom before execution, so the queue owner can release
and reload caches for a larger request or changed memory pressure.

The [official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3) describes
the joint audiovisual transformer and separate VAEs. The
[8 GB benchmark installation](https://github.com/YHK-AI/MiniMax-H3-Benchmark/blob/main/docs/INSTALLATION.md)
demonstrates another implementation using a pruned INT8 denoiser, NVFP4 text
encoder and disk-backed loading. Kadan retains its supported, pinned INT8
bundle (68,972,509,501 download bytes); that benchmark's smaller bundle and
timings are not Kadan measurements. The
[MiniMax integration index](https://github.com/MiniMax-AI/awesome-minimax-h3-integration)
also lists NF4 and GGUF paths for other implementations. Those formats are not
enabled for Kadan H3. Existing checkpoint license notices and acknowledgment
remain in place.

Tier defaults and memory policy are covered by CPU unit checks. Generation
quality, speed, physical peak memory, audio synchronization and cancellation
during inference still require owner verification through the existing API queue
on each target machine. No inference or GPU load tests were run for this change.

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
protection or preserving residency across existing retirement boundaries.

All inference is executed by the API-owned worker. Developer generation commands
use `python bin/kadan.py infer` to submit requests through the live API and its
existing queue. There are no standalone model execution tools.
