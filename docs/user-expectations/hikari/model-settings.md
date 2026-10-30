# Model settings and downloads

Hikari's Settings lists the intended model for each capability, its download state and estimated storage use. Models missing from the connected workers are shown as unavailable. Users review any presented license notices and confirm before starting downloads. Downloads can wait, fail or remain incomplete; completion confirms acquisition, not that a model is loaded or a request will fit the machine.

The page shows each model's progress. Percentages are estimates; the completed state confirms success. **Resume downloads** retries incomplete downloads without downloading completed models again. Download sizes use verified totals when available; catalog estimates are labelled **estimated**, and unavailable sizes say **Size unknown**. These sizes describe downloaded files, rather than loaded memory or generated files.

Closing the page keeps accepted downloads running. After a failed or timed-out request, refresh model state before retrying because some downloads may already have started.

The settings page no longer offers hardware tier selection. Some existing Hikari workloads still depend on a previously selected tier, so a new installation may not be able to use those workloads until the backend migration is complete. Direct inference can also download a missing model, so the first request may take longer or fail during acquisition.

Downloaded checkpoints persist in their model store across ordinary container restarts. Deleting that store loses them. Hikari verifies downloads before treating a checkpoint as complete; users must not modify completed checkpoints externally or share one store between independent installations. With [remote workers](remote-workers.md), model files reside on the computers serving the workloads.

See [Model loading and context](model-lifecycle.md) for loaded-state behavior and [Settings architecture](../../architecture/hikari/model-settings.md) for download controls and saved settings.
