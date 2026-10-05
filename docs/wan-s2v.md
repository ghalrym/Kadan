# Wan2.2-S2V-14B

This checkpoint adds native speech-to-video conditioning through the official
WanS2V implementation. It requires an uploaded reference image and PCM WAV audio
at least as long as the requested video. Both inputs are resolved from Kadan's
managed upload store; URLs and caller-supplied server paths are not accepted.
The source image is cropped to the requested aspect ratio and audio is trimmed to
the requested duration. Output includes the uploaded speech, encoded as AAC.

The provider uses 16 fps, UniPC, shift 3, 40 sampling steps and guidance 4.5 from
the pinned S2V config. It does not invoke optional CosyVoice synthesis or prompt
extension. Native internal resolution follows Wan's aligned-area rules; frames
are resized to the requested display dimensions during encoding. Pose-video
conditioning is outside this checkpoint slice and is rejected.

The shared Kadan Wan worker owns loading, RAM/per-GPU admission, cancellation,
process cleanup and atomic publication. No ComfyUI runtime is used. Runtime code
is pinned to `1ea34ff48f87168174e12956e200b1d908b1c5ff`; checkpoint metadata is pinned
to `dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b`.

The approximately 45.8 GB selected bundle includes four transformer shards,
UMT5 encoder, Wan2.1 VAE, the UMT5 tokenizer and Wav2Vec2 audio encoder. Duplicate
Flax/PyTorch Wav2Vec representations, example videos and evaluation scripts are
excluded. Download-time metadata supplies exact byte counts and file digests.

Sources:
- https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/speech2video.py
- https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/configs/wan_s2v_14B.py
- https://huggingface.co/Wan-AI/Wan2.2-S2V-14B/tree/dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b

Tests use lightweight fixtures for conditioning, sampler arguments, WAV trimming,
encoding and failure propagation. No weights were downloaded, no real model was
run, and no worker environment was installed. Output quality, GPU compatibility,
peak memory and throughput remain unmeasured; no 3090 performance claim is made.
