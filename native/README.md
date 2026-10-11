# Native inference worker

The API uses one `kadan-inference-worker` process for chat, Decisions, image,
video, transcription and speech. Its native owner manages the FIFO, shared
resource ledger and engine lifecycle; Python provides API and preparation
transport. See `src/inference_worker.cpp` and `include/kadan/inference_owner.hpp`.

Build and install the `inference-runtime` component using the options in
[`api/Dockerfile`](../api/Dockerfile). The former standalone worker executables
and launchers have been removed. Engine libraries remain here; metadata inspection tool
sources live in `../bin/cpp/` and are exposed through `../bin/kadan.py native`.
Run `python bin/kadan.py native build` from the repository root to compile them.
Normal runtime builds leave `KADAN_BUILD_TOOLS` disabled.

All inference is executed by the API-owned worker. Developer generation commands
use `python bin/kadan.py infer` to submit requests through the live API and its
existing queue. There are no standalone model execution tools.
