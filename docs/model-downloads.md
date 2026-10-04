# Local checkpoint storage

Settings uses `/v1/models` to download and select only the three curated checkpoints.
Selection does not load a model or imply that GPU inference has been validated.
The catalog pins full Hugging Face commit IDs; review links and license metadata
are in `api/services/model_catalog.py`. Qwen and GPT-OSS are Apache 2.0; GLM is MIT.
Upstream model cards remain authoritative for additional usage conditions.

Set `KADAN_MODEL_DIR` to a writable local directory with sufficient space
(default `~/.local/share/kadan/models`). Persist this directory across container
recreation. Run **one API worker**; selection and runtime coordination are in
process. The downloader additionally takes a filesystem advisory lock to reject
accidental concurrent writers from another process. Do not share the store
between independently running API servers. No Hugging Face credentials are needed
for these public checkpoints, and the service does not execute repository code.

Docker Compose mounts the named `model_data` volume at `/var/lib/kadan/models`.
It survives service/container recreation; `docker compose down -v` removes it.
Use a bind mount instead if managing model storage on a particular host disk.

Only root safetensors and listed tokenizer/config assets are downloaded. GPT-OSS
`original/` and `metal/` are excluded to avoid redundant formats. GLM's root MTP
weights are included. Before downloading, upstream pinned metadata supplies exact
sizes and SHA-256 (LFS) or Git SHA-1 (regular files); insufficient free space stops
the operation. Every streamed file is verified and the weight index must reference
only downloaded files. Completion is published by an atomic directory rename.
Selection is persisted separately and only complete checkpoints can be selected.
A completed checkpoint is never overwritten or deleted through this API.

Downloads are serialized. Progress reports bytes verified/in transfer, and HTTP
errors or failed integrity checks never produce a complete checkpoint. Cancel is
cooperative between 1 MiB reads (network reads have a 30-second timeout); UI shows
`cancelling` until cleanup finishes. Retry starts from scratch, deleting only the
application-owned staging directory. On API restart, interrupted staging is not
considered complete and the next Download restarts it. Error/progress counters are
in memory; completed files and selection persist. A manual disk modification can
invalidate a stored model; completeness rechecks size on selection, while hashes
are checked during download. Do not edit checkpoint files externally.

The runtime reserves selection through `model_manager.acquire_runtime_model()`
and releases it on unload/failure with `release_runtime_model()`. Selection
changes receive HTTP 409 while this lease is held. The returned catalog entry and
absolute completed path are the only supported runtime model inputs.

Tests stream tiny controlled bytes and never download real weights. Network access
to Hugging Face, several hundred GB of disk, and GPU inference still need local
validation; this cloud test does not establish performance or memory fit.
