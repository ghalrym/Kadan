# Language-model loading and context

Inference normally loads its language model on demand. API clients can also read /model-lifecycle, explicitly load through /model-lifecycle/load or unload through /model-lifecycle/unload. Reading status does not start inference or load a model. Unloaded, loading, ready, offloaded, unloading and error are distinct reported states; a saved tier or a successful status read is not proof that a model is resident.

An explicit load uses the selected tier's language model unless a supported model identity is supplied. It shares the normal queue, preparation and memory admission, so it can wait or fail. Its HTTP 202 response is not a guarantee that the requested model and context will fit future prompts. Native preparation can acquire missing checkpoints; loading is not guaranteed to be download-free. A different supported model must not be silently substituted to satisfy a request.

The default saved context is 65,536 tokens per language checkpoint. PUT /v1/models/{model_id}/context saves a positive integer limit or null to use that checkpoint's architecture maximum on load. A boolean is not an integer setting. An excessive limit is rejected when the architecture maximum is known; loading also validates it against the actual checkpoint. Selecting the architecture maximum does not guarantee enough physical memory to use it.

For /model-lifecycle/load, omitting context_limit retains the saved/default value. Explicit null requests the architecture maximum for that load; a supplied positive integer overrides the saved value for that load. This does not persist a new default. Saving a context setting is a separate operation and does not prove that a resident model has already been resized.

Status distinguishes configured, effective and supported context limits. Context is a token capacity, not a character count or a promise to generate that many output tokens. The current language reply cap is 256 output tokens. Requests that exceed the admitted context must fail explicitly rather than silently truncating conversation history.

Explicit unload requests cancellation of pending language work and waits for its owned cleanup. It is not a control for cancelling other inference types or deleting downloaded checkpoints. Later inference can load the model again. Automatic model expiry and reusable-prefix retention have separate rules in [Inference operations](inference.md).
