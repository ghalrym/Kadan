# Wan2.2 I2V-A14B

This checkpoint generates video conditioned on an uploaded image. The API requires `image_id`; it does not treat an image-to-video model as text-only. Conditioning is decoded from Kadan-owned uploaded storage, converted to RGB and fitted to the requested output aspect before native generation.

The [official original-format checkpoint](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/tree/206a9ee1b7bfaaf8f7e4d81335650533490646a3) is pinned to `206a9ee1b7bfaaf8f7e4d81335650533490646a3`. It includes both high- and low-noise experts, UMT5 weights and tokenizer, and the Wan2.1 VAE. The official tree reports approximately 126 GB; the downloader verifies exact file sizes and hashes before publishing. The duplicate Diffusers repository, examples and repository code are excluded.

The native worker uses Wan2.2 source `1ea34ff48f87168174e12956e200b1d908b1c5ff` and its `WanI2V` class. It retains the official 40-step UniPC configuration, `(3.5, 3.5)` expert guidance and checkpoint boundary, with shift 3 for 480p and 5 for 720p. Output is 16 fps. Kadan owns the isolated worker and all memory leases until process exit, including cancellation. No ComfyUI service is involved.

Dependencies: native Wan engine and its media input/job infrastructure. Tests use synthetic images and injected native factories/processes; real model weights, CUDA inference, visual quality and performance have not been tested. No throughput claim is made for a 3090 or other GPU. Code and tests do not install a worker or download weights.
