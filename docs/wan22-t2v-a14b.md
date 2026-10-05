# Wan2.2 T2V-A14B

This checkpoint slice connects original Wan2.2 T2V-A14B text-to-video to the shared
native Wan engine and Kadan video queue. It is a silent-video model, not the
speech-to-video or animation checkpoint. Dependencies are the shared Wan native
engine, video jobs API, and component-aware catalog/download service.

The official repository is `Wan-AI/Wan2.2-T2V-A14B`, pinned at
`c8c270b13ee05bfa474194ac9fb07a5868a97cea`, under Apache-2.0. Its approximately
126 GB original bundle includes both `high_noise_model` and `low_noise_model`
(each six safetensors shards plus config/index), `models_t5_umt5-xxl-enc-bf16.pth`,
`Wan2.1_VAE.pth`, and the `google/umt5-xxl` tokenizer. Only the two exact published
`.pth` files are allowed; arbitrary pickle files and repository code are excluded.
The native package supplies executable code. Downloads retain per-file digest
verification, disk preflight, staging, cancellation and atomic publication.

The official native task is `t2v-A14B`: 16 fps, 40 sampling steps, shift 12,
low/high-noise guidance `(3.0, 4.0)`, and expert boundary 0.875. The supported
published canvases are 1280×720, 720×1280, 832×480 and 480×832. This checkpoint does
not promise square output or 1080p. The Wan engine owns the frame-count contract
and holds Kadan's shared memory reservations through worker cleanup.

No weights were downloaded and no GPU inference was run. Tests use tiny local
byte fixtures and native API mocks; they do not establish output quality, peak
memory, latency, or 3090 throughput. Runtime installation and deployment remain
separate actions.

Sources:
- [Official checkpoint and immutable revision](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B/tree/c8c270b13ee05bfa474194ac9fb07a5868a97cea)
- [Official native task configuration](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/configs/wan_t2v_A14B.py)
