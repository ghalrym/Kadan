# Inference worker

The API uses one `kadan-inference-worker` process for chat, Decisions, image,
video, transcription and speech. It owns the FIFO, checkpoint acquisition,
input preparation, engine lifecycle, result packaging and workspace cleanup.
The API submits serialized Pydantic tasks and relays events and results.

`src/inference_request.cpp` decodes the transport envelope into named C++ requests;
`include/kadan/inference_owner.hpp` processes the queue head. Each modality's
`src/inference_*_request.cpp` owns its preparation and response formatting alongside
its engine. Chat messages, transcription PCM and typed decision results pass directly
in memory. Decision answers are formatted as JSON once for the response.
`JobWorkspace` owns only workspace paths and cleanup.

`src/model_store.cpp` is the single checkpoint downloader for both inference and
Settings download requests. It also owns checkpoint validation, context limits and
model selection. The worker publishes status snapshots at startup and after tasks;
the API serves those snapshots, observes download progress and forwards cancellation.
Settings changes use typed queue tasks.
`src/whisper_export.cpp` and `src/tokenizer_assets.cpp` convert checkpoint metadata
in C++. The worker never starts a Python interpreter.
Cancellation finishes owned work and releases the workspace before the next job.

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
