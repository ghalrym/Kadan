# Language-model loading and context

Hikari runs Inkling 2 Mini (`inkling-2-mini-iq4-nl` by default, or `inkling-2-mini-q8-0` via `HIKARI_LLM_QUANT`) on the native Strata worker (`hikari-worker-llm`). It loads on demand when a local chat or screenshot request runs, stays resident while memory permits, and yields or unloads automatically when idle or when another worker needs GPU memory. Users do not preload it manually or edit context limits in the web UI; context size is configured via `HIKARI_LLM_CONTEXT_LIMIT` (default `65,536` tokens). Download the verified checkpoint after reviewing its licenses before submitting local inference, or switch Language to a connected Claude or Codex subscription.

Status reads do not load the model. They distinguish an idle, unloaded worker from active or resident language work and report unavailable workers, incomplete downloads and unconfirmed cleanup as errors. A completed download establishes file readiness, not successful inference or physical fit.

When language work is loading, Hikari reports loading. Once loaded and generating or resident, it reports ready, allowing Kadan to show the agent's active response. After explicit unload or idle/memory-pressure eviction completes cleanup, it returns to unloaded.

The configured context limit includes the complete prompt (including any encoded screenshot tokens) and reply. Generation can use the remaining context until completion or cancellation. Requests leaving no reply space or exceeding available memory fail without truncating history or substituting another model.

Explicit unloading cancels pending language work and waits for cleanup before releasing its resources. It does not cancel other inference types or delete downloaded models. Later inference loads the model again.

See [Inference operations](inference.md) for scheduling, GPU sharing, and prefix-cache reuse.
