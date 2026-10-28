# Model settings and downloads

Hikari's Settings lets users choose a hardware tier and download its models. A new installation has no selected tier. The suggestion does not select it or start downloads; users review the model licenses and confirm first.

The saved tier supplies models for features still using tiers, including calls, and video defaults. Hikari Chat and Kadan text conversations use LFM separately. Downloads can wait, fail or remain incomplete. A selected tier or completed download does not mean a model is loaded, or that every request will fit the machine.

The download dialog shows each model's progress. Percentages are estimates; the model's completed state confirms success. **Resume downloads** retries incomplete downloads without requiring completed models to be downloaded again.

Download sizes use verified checkpoint totals when available. Catalog estimates are labelled **estimated**; unavailable sizes say **Size unknown**. Completed native checkpoints retain their verified totals across ordinary worker restarts. These sizes describe downloaded model files, rather than loaded memory or generated files.

**Continue in background**, closing the dialog or leaving Settings keeps downloads running. Tier selection is disabled while its downloads are active. **View downloads** reopens progress. After a failed or timed-out settings change, refresh the saved state before retrying.

Direct inference can also download a missing model, so the first request may take longer or fail during acquisition. Changing tiers can unload the previous language model after earlier chat work finishes. It does not delete downloaded models or save playground conversations.

Downloaded checkpoints persist in their model store across ordinary container restarts. Deleting that store loses them. Hikari verifies downloads before treating a checkpoint as complete; users must not modify completed checkpoints externally or share one store between independent installations. With [remote workers](remote-workers.md), model files reside on the computers serving the workloads.

Settings also shows individual downloads for connected LFM3 workers. The standard deployment enables conversation and screenshot workers; their downloads are listed under **Experimental models** during the migration. Each model has its own license link, code notice, state and progress. Opening its dialog does not accept the license or start a download: users must check the acceptance box and confirm. These downloads do not change the selected hardware tier. A failed or timed-out request requires refreshing model state before retrying, because the download may already have started. Completed downloads confirm local acquisition, not a loaded model or successful inference.

See [Model loading and context](model-lifecycle.md) for loaded-state behavior and [Settings architecture](../../architecture/hikari/model-settings.md) for download controls and saved settings.
