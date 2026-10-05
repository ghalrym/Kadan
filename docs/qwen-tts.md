# Native Qwen3-TTS

The shared engine depends on the speech API PR. The first checkpoint PR extends
the download catalog for component bundles without enabling another model.
Individual checkpoint PRs enable each of the five official models separately.
The engine exposes only enabled choices, and requires a complete, revision-matching
checkpoint from Kadan's download manager. It never downloads on generation.

Qwen's pinned package requires transformers 4.57.3 and accelerate 1.12.0, which
conflict with the main LLM runtime. Provision a separate Python environment using
`api/requirements-qwen-tts.txt`, then set `KADAN_QWEN_TTS_PYTHON` to its Python
executable. This PR does not install that environment. The API owns the child
process directly; there is no separate model server. `KADAN_QWEN_TTS_DEVICE`
defaults to `cuda:0`; `cpu` and other explicit CUDA indices are supported. CPU uses
float32, CUDA bfloat16, and SDPA attention. There is no hardware speed claim.

Every request reserves conservative host and per-device memory through the same
ResourceManager as chat, before launching the child. Generation holds an active
lease. Disconnect, timeout and failure reap the process before releasing memory
accounting. This initial implementation unloads after every request; it does not
claim warm residency performance. Admission estimates are not measured peak usage.

The API returns complete PCM WAV audio as base64; the upstream `generate_*`
methods do not yield streaming audio. Uploaded clone bytes remain in temporary
request storage and are deleted after the child exits. Clone requests require a
reference transcript unless speaker-only mode is selected. No remote sample URLs
or server paths are fetched. Generated results are kept in the current browser;
server history remains empty.

Verified sources:
- https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/tree/0c0e3051f131929182e2c023b9537f8b1c68adfe
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/tree/5ecdb67327fd37bb2e042aab12ff7391903235d3
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/fd4b254389122332181a7c3db7f27e918eec64e3
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc

Checkpoint files include root config, generation_config, tokenizer_config,
preprocessor_config, merges.txt, vocab.json and model.safetensors, plus the bundled
speech_tokenizer config.json, configuration.json, preprocessor_config.json and
model.safetensors. Both weights are required; no separate tokenizer download or
shard index is assumed. Official repositories specify Apache-2.0.

Validation uses synthetic audio, fake workers and API/browser fixtures. No model
weights were downloaded, no actual Qwen inference was run, and CUDA behavior,
quality and measured memory/latency remain unverified.
