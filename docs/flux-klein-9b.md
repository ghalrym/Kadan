# FLUX.2 klein 9B

Adds `black-forest-labs/FLUX.2-klein-9B` at immutable revision `92196c8e11f7b6cf2b7493e037d8c5345c559216`, identified by the official public README commit link. The complete native bundle is approximately 34.8 GB, including four text-encoder shards, two transformer shards, VAE, tokenizer and scheduler. The duplicate root export is excluded.

The native generation/editing recipe uses four steps and guidance 1.0 through the pinned `Flux2KleinPipeline`. It reuses Kadan's image selection, leases, offload, cancellation and PNG delivery. No hardware-based checkpoint exclusions are applied.

This checkpoint is gated under the FLUX non-commercial license and usage conditions. The download notice requires explicit Continue and links the official license. Download acknowledgement neither grants commercial rights nor accepts Hub terms. The deployment must have approved Hub access and an existing authorized `HF_TOKEN`; the code only reads this variable when requesting a gated checkpoint, never stores it or sends it to redirected CDN hosts. No credentials or license approvals were created during implementation. Deployers must review all license obligations, including the model card's input/output filters or manual review requirements.

Tests use fixture pipelines and mocked HTTP requests. No weights, live authenticated download, GPU inference, quality or throughput tests were run. Public file previews and README links establish artifact names/revision; full gated contents were not fetched in development.

Source: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
