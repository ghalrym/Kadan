# MiniMax H3 FL2VA download

This slice adds one checkpoint to Settings: MiniMax H3 FL2VA, pinned to
`42ed227ee7df40d41602854ae760620d6eb651fe` in `MiniMaxAI/MiniMax-H3`.
The [official original-format bundle](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA)
is approximately 144 GB. Exact bytes and integrity digests are obtained from the
pinned metadata before downloading; disk admission requires those bytes plus
1 GiB. The root index, FL2VA index, original processor/tokenizer, text encoder,
transformer, audio VAE and video VAE/source weights are included. Duplicate root
Diffusers weights, Ref2VA and repository Python implementations are excluded.
Native implementations must supply their own reviewed code; no remote code is run.

Download opens one notice with the [official license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE)
and Cancel/Continue. The license excludes use in the US, EU, UK and South Korea,
including personal use. Acknowledgement is not a rights grant. Each retry asks
again; the API rejects a download without explicit acknowledgement.

This PR is download-only. The separate native FL2VA text-to-video/audio provider
PR depends on the complete local bundle and shared video job engine. No native
inference, Ref2VA, Turbo adapters, H3-Context-IR or 2K regeneration is claimed here.

Downloads use the existing writer lock, cancellation, digest verification and
atomic publish. Nested indexes are checked relative to their component directory.
`get_checkpoint(model_id)` returns a complete immutable directory without changing
LLM selection; providers own RAM/VRAM admission and cleanup separately.

Verification uses synthetic bytes only. Backend fixtures cover required
components, nested indexes, symlinks, integrity and language-model separation.
`frontend/tests/h3-download.browser.mjs` checks desktop/mobile Cancel, Escape,
Continue, repeated clicks, failure, retry and cancellation with mocked HTTP.
No H3 weights were downloaded, native runtime installed, or GPU inference run.
