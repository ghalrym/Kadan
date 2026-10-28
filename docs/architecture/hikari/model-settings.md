# Model settings

Hikari uses fixed assignments from `ModelAssignments`, returned with observed checkpoints by both Settings and the worker model snapshot. There are no hardware tiers or saved tier selection. The API aggregates actual worker catalogs; an assigned but disconnected LFM worker contributes no fabricated checkpoint or download status.

`GET /v1/settings` returns `models`, `defaults`, `video_defaults` and `whisper_formatting`. `PUT /v1/settings` changes only the optional formatting preference. Video defaults are five seconds at 480p, independent of hardware; explicit request settings remain subject to actual admission.

Each model is downloaded through its existing model-specific action with explicit license acknowledgement. Context configuration goes to the worker owning the language checkpoint. Its saved per-model setting is stored separately from the pinned artifact manifest and affects future requests only; null uses the architecture maximum. The LFM executor still loads per request and retains no resident model or prefix cache.

The native model store no longer reads or writes `tier.json`. Existing checkpoint directories and historical settings files are left untouched. LFM snapshots retain their existing model-volume paths so this migration does not lose verified downloads.
