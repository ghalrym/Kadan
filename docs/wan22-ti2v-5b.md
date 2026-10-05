# Wan2.2 TI2V-5B

This checkpoint adds native text-to-video and optional first-image conditioning,
using `wan.WanTI2V`. It depends on the shared Wan worker and video job engine.
Only Wan2.2-TI2V-5B is enabled by this PR; other Wan checkpoints are separate PRs.

Pinned sources:
- https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e

The approximately 34.2 GB original bundle contains three transformer shards,
Wan2.2 VAE, UMT5 encoder and its tokenizer. Example media and duplicate formats
are excluded. Apache-2.0 applies. Required `.pth` components are explicitly named
and receive the same manifest hash verification as safetensors downloads.

The pinned native configuration uses 50 UniPC steps, shift 5, guidance 5 and
24 fps. Its nominal 720p preset is 1280×704 (or 704×1280 portrait). Input images
are fitted to the selected preset before conditioning; generated alignment frames
are trimmed to the requested duration. No synthetic audio is added.

Kadan admits RAM/VRAM before starting an offline isolated worker and retains the
reservation through process cleanup. The operator configures `KADAN_WAN_PYTHON`
for an environment prepared from `api/requirements-wan.txt`; no installation or
weight download occurs on inference. Full GPU execution, memory peaks and speed
have not been measured. No model weights were downloaded or run for this PR.

Tests cover exact native arguments, real CPU tensor frame conversion with a mocked
model, atomic synthetic downloads, HTTP upload-ID resolution, and rendered desktop
and mobile upload failure/retry. Inputs and jobs are local to this deployment;
job metadata is not restored after restart. Uploaded media is stored locally and
not sent to a hosted inference provider.
