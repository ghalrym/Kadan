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

Build and install the `inference-runtime` component using the options in
[`api/Dockerfile`](../api/Dockerfile). The former standalone worker executables
and launchers have been removed. Engine libraries remain here; metadata inspection tool
sources live in `../bin/cpp/` and are exposed through `../bin/kadan.py native`.
Run `python bin/kadan.py native build` from the repository root to compile them.
Normal runtime builds leave `KADAN_BUILD_TOOLS` disabled.

All inference is executed by the API-owned worker. Developer generation commands
use `python bin/kadan.py infer` to submit requests through the live API and its
existing queue. There are no standalone model execution tools.
