# Wan2.2 Animate-14B

This checkpoint adds native character animation and replacement using a reference
image and driving MP4. It depends on the shared Wan worker and opaque input-upload
API. Kadan admits the complete job under its shared RAM/VRAM lease, including
preprocessing, generation, encoding and process-group cleanup.

The original checkpoint is `Wan-AI/Wan2.2-Animate-14B` at
`cb93a225fbaf1ca100f54e79da8f994995b689b3`. The model card declares Apache-2.0.
The selected bundle is approximately 58 GB: four original transformer shards,
UMT5 and CLIP weights/tokenizers, VAE, the original relighting LoRA, ViTPose and
YOLO ONNX models, and SAM2 large weights. It excludes duplicate XLM-R weights,
converted relighting LoRA, unused SAM2 sizes, sample media, repository scripts
and FLUX. The downloader obtains exact sizes and verifies each digest before
atomic publication.

The Kadan worker first center-crops the driving video and reference image to the
requested landscape or portrait 720p canvas, trims to the requested duration and
uses 30 fps. A shorter source video fails explicitly. The pinned official
`preprocess_data.py` performs pose/face extraction, basic pose retargeting in
animation mode, and background/mask extraction in replacement mode. These are
real pipeline calls, not placeholders or an assumption that raw video is already
pose conditioning. FLUX-based enhanced retargeting is disabled.

Generation uses native `WanAnimate`, its 77-frame recurrent windows, 20 steps,
shift 5, guidance 1 and UniPC sampler. Replacement enables the original relighting
LoRA. Prompt, negative prompt and seed are forwarded. Output is silent MP4;
requested frame count and canvas are checked. Parent-owned temporary storage is
removed even if the child is cancelled while processing inputs.

The official preprocessing targets a single person. Basic retargeting works best
with front-facing, extended reference poses; it does not guarantee correct masks
or motion for crowded/occluded scenes. Input preprocessing, segmentation quality,
model outputs, memory peaks, timing and 3090 compatibility have not been verified
on actual weights or GPU hardware. Tests use small fixtures and native API mocks.

`api/requirements-wan-animate.txt` defines the optional isolated environment,
including the same SAM2 source commit used by the official project. Set
`KADAN_WAN_PYTHON` to that environment's interpreter. No environment installation,
weights download, or deployment is performed automatically. Dependency imports
and FFmpeg/preprocessing are confined to the Kadan-owned child process group.

Sources:
- [Immutable original checkpoint](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/tree/cb93a225fbaf1ca100f54e79da8f994995b689b3)
- [Official preprocessing guide](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/animate/preprocess/UserGuider.md)
- [Native Animate implementation](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/animate.py)
