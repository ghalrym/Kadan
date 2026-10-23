# Model settings and downloads

Hikari separates saved tier selection, checkpoint acquisition and model residency. Persisting a choice must not imply that its download or load succeeded. See the privately provided expectations for model settings and downloads, and [Model lifecycle](model-lifecycle.md) for loading and context settings.

`PUT /v1/settings` saves the tier and queues incomplete checkpoints. Omitted settings retain their values; a null tier does not clear the selection. Transcription formatting is an independent saved switch with a per-request override, so changing tiers alone does not toggle it. Settings and model-status reads do not run inference.

`POST /v1/models/{model_id}/download` queues acquisition, and `DELETE` on the same path requests cancellation. HTTP 202 acknowledges the operation rather than completed downloading or immediate resource release. Clients poll model status for the outcome. A second active download for the same checkpoint, or cancellation with no active download, is a conflict. Cancellation does not delete a completed checkpoint.

Native request preparation can acquire a missing checkpoint. An unusable existing directory or acquisition failure remains an error. Kadan separately gates ordinary chat sending on its reported downloaded-language-model state. These entry points therefore have different prerequisites without changing model ownership.

Progress is reported per checkpoint. Aggregate percentages may use catalog estimates before byte totals are known; checkpoint state establishes completion. Tier changes can unload the prior resident language model after earlier chat work finishes, but do not delete its stored checkpoint.

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
