# Model settings and downloads

Hikari uses one fixed model stack rather than hardware tiers: LFM conversation, screenshot understanding and speech, Qwen Image Turbo for image generation and editing, H3 for video, Whisper medium for transcription, and Laya for decisions. Transcript formatting uses the conversation model. The fixed assignments do not guarantee that every workload fits every computer.

Settings shows each model's observed download state and license. A worker that is not connected is shown as unavailable; a configured assignment does not fabricate a downloaded checkpoint. Users review each model's license and code notice, check acceptance and confirm before downloading. Opening a dialog or starting a container does not accept a license. Inference requires a completed local download; a request for a missing or incomplete model fails and asks the user to download it in Settings.

Downloads can wait, fail or remain incomplete. Progress percentages are estimates; the completed state confirms local acquisition, not a loaded model or successful inference. Cancel requests wait for worker cleanup. A cancelled or failed download can be retried, and completed models do not need downloading again. After an uncertain request, refresh model state before retrying because the operation may already have started.

Leaving Settings does not cancel an accepted download. Downloaded checkpoints persist across ordinary container restarts; deleting their model store loses them. Users must not modify verified checkpoints externally or share a store between independent installations. With [remote workers](remote-workers.md), model files reside on the computers serving the workloads.

Language context can be saved per model or set to the checkpoint's maximum. See [Model loading and context](model-lifecycle.md) for context and execution limits.
