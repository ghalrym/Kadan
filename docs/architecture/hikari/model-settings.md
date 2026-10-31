# Model settings and downloads

Hikari separates fixed capability assignments, checkpoint acquisition and model residency. `ModelAssignments` defines the API's assignments; the frontend presents the same fixed capabilities. The native catalog retains explicit MoE checkpoint metadata for the retained engine, but these entries are not exposed as model choices. See [model expectations](../../user-expectations/hikari/model-settings.md).

`GET /v1/settings` returns fixed assignments, observed checkpoints, five-second/480p video defaults and the saved transcription formatting switch. `PUT` changes only formatting. Tier selection, Whisper selection, manual language loading and public saved-context controls are removed. Old tier and Whisper selection files are ignored without deleting or rewriting them. Existing checkpoint directories and volumes are untouched.

Worker protocol 5 rejects older workers whose settings and preparation still depend on tiers. Fixed language, vision and audio calls use their separate pinned Python stores; native media, Qwen speech, Whisper, S1-mini and decisions retain the native store. The API exposes a fixed-assignment checkpoint only while its owning worker is connected. A native store can report every catalog entry on its volume without being able to execute each one, so observations are filtered against the reporting worker's type before they enter the combined model list. Duplicate owner observations are copied once; primary native settings remain authoritative.

Whisper-formatting writes and synchronization target connected native persistence stores, including retained language, speech and formatting workers. LFM workers acknowledge configure commands without persisting this native preference, so they cannot confirm a settings save or receive irrelevant native synchronization. With no native persistence owner, saving fails explicitly. Model files, volume identity, normal request admission and inference routing are unchanged.

`POST /v1/models/{model_id}/download` requires explicit `license_acknowledged: true` and queues acquisition. `DELETE` on the same path requests cancellation. HTTP 202 acknowledges the operation rather than completion or immediate resource release. Clients poll model status for the outcome. A second active download for the same checkpoint, or cancellation with no active download, is a conflict. Cancellation does not delete a completed checkpoint. Unknown or unavailable checkpoints fail before queueing.

Inference preparation requires a verified local checkpoint and never initiates acquisition. GGUF preparation retains its metadata validation and derived-asset checks; image encoders/tokenizers, Qwen speech tokenizer assets, Whisper exports and the native image/video runtime remain in place. Missing or unusable files produce an error. Formatting remains independent, including per-request overrides and raw-transcript recovery.

Progress is reported per checkpoint. Aggregate percentages may use catalog estimates before byte totals are known; checkpoint state establishes completion.

Completed native status reports sum the downloaded artifact sizes recorded in a validated completion manifest. Existing layout-specific checks remain, including pinned composite/GGUF file lists and single-file revisions. Both downloaded and total bytes use that sum; invalid or incomplete stores do not receive completed totals. Markers and derived caches are excluded. Settings uses observed totals first, labels catalog fallbacks as estimates, and identifies missing sizes explicitly.

## Checkpoint storage

The catalog pins revisions and verifies downloaded files before publishing them.
Docker Compose persists models in `model_data` at `/var/lib/kadan/models`.
`docker compose down -v` deletes named volumes, including stored data.
Do not modify completed checkpoints externally or share a store between
independent API servers. Review the catalog model cards and licenses before use.

Cancellation joins checkpoint preparation and native cleanup. Memory reservations
are conservative admission estimates, not physical hard limits; other programs
can change available memory. A cleanup failure stops admission rather than
reusing uncertain allocations. Recovered model directories can remain on their
existing storage; copying the entire collection is unnecessary.
