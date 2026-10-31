# Language-model loading and context

Hikari runs the fixed LFM3 1.7B language checkpoint in a separate worker. It loads for each request and releases the model after completion. Users cannot select another checkpoint, preload it manually, or save a different context limit. Download the verified checkpoint after reviewing its licenses before submitting inference.

Status reads do not load the model. They distinguish an idle, unloaded worker from active language work and report unavailable workers, incomplete downloads and unconfirmed cleanup as errors. A completed download establishes file readiness, not successful inference or physical fit.

The model's 262,144-token context includes the complete prompt and reply. Generation can use the remaining context until completion or cancellation. Requests leaving no reply space or exceeding available memory fail without truncating history or substituting another model.

Explicit unloading cancels pending language work and waits for cleanup before releasing its resources. It does not cancel other inference types or delete downloaded models. Later inference loads the model again.

See [Inference operations](inference.md) for scheduling and the retained native engine's separate prefix behavior.
