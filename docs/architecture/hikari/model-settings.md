# Model settings and downloads

Hikari separates fixed capability assignments, checkpoint acquisition and model residency. `ModelAssignments` defines the API's assignments; the frontend presents the same fixed capabilities. Language (including multimodal screenshot vision and single-pass decisions) uses the active Inkling 2 Mini checkpoint (`inkling-2-mini-iq4-nl` by default, or `inkling-2-mini-q8-0` via `HIKARI_LLM_QUANT`, and the abliterated `inkling-2-mini-abliterated-*` variants via `HIKARI_LLM_ABLITERATED=true`). See [model expectations](../../user-expectations/hikari/model-settings.md).

`GET /v1/settings` returns fixed assignments and observed checkpoints. `PUT` updates the active local Inkling 2 Mini variant (`llm_model`, persisted in `llm-selection.json` across native stores with `HIKARI_LLM_QUANT` / `HIKARI_LLM_ABLITERATED` as fallback defaults). Tier selection, Whisper selection, transcript formatting, manual language loading and public saved-context controls are removed. Old tier and selection files are ignored without deleting or rewriting them. Existing checkpoint directories and volumes are untouched.

Worker protocol 5 rejects older workers whose settings and preparation still depend on tiers. All four local workloads — native Inkling 2 Mini language (including vision and decisions), FLUX.2 Klein 4B, Qwen3-TTS speech, and Qwen3-ASR 1.7B transcription — are Apache 2.0 licensed and share the native store. The API exposes a checkpoint only while its owning worker is connected. A native store can report every catalog entry on its volume without being able to execute each one, so observations are filtered against the reporting worker's type before they enter the combined model list. Duplicate owner observations are copied once; primary native settings remain authoritative.

`llm_model` selection writes and synchronization target all connected native persistence stores (`llm`, `tts`, `speech`, `image`). With no native persistence owner, saving fails explicitly. Model files, volume identity, normal request admission and inference routing are unchanged.

`POST /v1/models/{model_id}/download` requires explicit `license_acknowledged: true` and queues acquisition. `DELETE` on the same path requests cancellation. HTTP 202 acknowledges the operation rather than completion or immediate resource release. Clients poll model status for the outcome. A second active download for the same checkpoint, or cancellation with no active download, is a conflict. Cancellation does not delete a completed checkpoint. Unknown or unavailable checkpoints fail before queueing.

Settings starts downloads either for every capability (the summary button) or for one row, and submits only the idle, incomplete checkpoints among them through the per-model download endpoint. Confirmation sizes and license notices use that same set. Other active downloads do not block starting additional checkpoints; failed submissions require a fresh status read before retrying because earlier requests may have been accepted.

Inference preparation requires a verified local checkpoint and never initiates acquisition. GGUF preparation retains its metadata validation and derived-asset checks; image encoders/tokenizers, Qwen3-TTS and Qwen3-ASR tokenizer assets, and the native image runtime remain in place. Missing or unusable files produce an error.

Progress is reported per checkpoint. Aggregate percentages may use catalog estimates before byte totals are known; checkpoint state establishes completion.

Completed native status reports sum the downloaded artifact sizes recorded in a validated completion manifest. Existing layout-specific checks remain, including pinned composite/GGUF file lists and single-file revisions. Both downloaded and total bytes use that sum; invalid or incomplete stores do not receive completed totals. Markers and derived caches are excluded. Settings uses observed totals first, labels catalog fallbacks as estimates, and identifies missing sizes explicitly.

## Checkpoint storage

The catalog pins revisions and verifies downloaded files before publishing them.
Kadan stores downloaded models inside `<data_dir>/models` (or `KADAN_MODEL_DIR`).
Do not modify completed checkpoints externally or share a store between
independent API servers. Review the catalog model cards and licenses before use.

Cancellation joins checkpoint preparation and native cleanup. Memory reservations
are conservative admission estimates, not physical hard limits; other programs
can change available memory. A cleanup failure stops admission rather than
reusing uncertain allocations. Recovered model directories can remain on their
existing storage; copying the entire collection is unnecessary.
