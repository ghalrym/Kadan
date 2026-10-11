# Native inference worker

The API uses one `kadan-inference-worker` process for chat, Decisions, image,
video, transcription and speech. Its native owner manages the FIFO, shared
resource ledger and engine lifecycle; Python provides API and preparation
transport. See `src/inference_worker.cpp` and `include/kadan/inference_owner.hpp`.

Build and install the `inference-runtime` component using the options in
[`api/Dockerfile`](../api/Dockerfile). The former standalone worker executables
and launchers have been removed. Engine libraries and separate diagnostic tools
remain available.
