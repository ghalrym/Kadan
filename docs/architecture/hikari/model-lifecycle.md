# Language-model lifecycle and context

The public lifecycle describes the native `hikari-worker-llm` Strata worker running Inkling 2 Mini (`inkling-2-mini-iq4-nl` by default, or `inkling-2-mini-q8-0` via `HIKARI_LLM_QUANT`). See [language lifecycle expectations](../../user-expectations/hikari/model-lifecycle.md).

`GET /model-lifecycle` reports the current residency state (`unloaded`, `loading`, or `ready`) of the connected `llm` worker alongside the active Inkling 2 Mini checkpoint and effective context limit (`HIKARI_LLM_CONTEXT_LIMIT`, default `65536`). Missing workers, incomplete checkpoints and unconfirmed cleanup report an error. A download does not establish successful inference.

`POST /model-lifecycle/unload` cancels owned language jobs, joins cleanup, and requests confirmed native offload before returning status. Physical charges remain until release is confirmed. It neither removes checkpoint files nor cancels media work. Public manual loading and saved-context endpoints remain retired; context size is configured via `HIKARI_LLM_CONTEXT_LIMIT` on `hikari-worker-llm`.

Placement, memory admission, prefix-cache reuse, and GPU sharing with other workers (such as Qwen-Image and H3 video) are described in [Language worker](worker-llm.md) and [Inference architecture](inference.md).
