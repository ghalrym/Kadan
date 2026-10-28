# Language-model lifecycle and context

The fixed LFM conversation model loads per request after shared memory admission. Completed and cancelled requests release their executor, model and cache; there is no resident language engine or retained conversation prefix between requests. Download readiness remains available from the observed model catalog and is distinct from active execution.

`GET /model-lifecycle` reads status without starting work. Current execution reports loading while language work is outstanding and unloaded after it settles. It reports the fixed model ID, observed saved context and architecture maximum; no effective resident context is fabricated. A missing language worker is reported as unavailable. The retained schema's other historical state labels do not promise native resident-engine behavior.

`POST /model-lifecycle/load` is unavailable for this executor and fails explicitly rather than pretending to preload. `POST /model-lifecycle/unload` cancels queued and active conversation work, waits for cleanup and releases the worker grant. It does not cancel other inference types or delete model files.

`PUT /v1/models/{model_id}/context` saves a positive integer no larger than the checkpoint maximum; null selects that maximum. The owning worker stores this setting outside the immutable artifact manifest. Future queued requests resolve the saved value when they begin execution. Changing it does not resize an executor already running, and replicas synchronize their observed per-model settings.

Context is the combined prompt and reply budget in tokens. A generation uses the available space after the complete prompt until EOS, cancellation or its requested context boundary; it does not discard history to fit. An architecture-supported window can still exceed physical resources. The [language worker](worker-llm.md) and [shared scheduling](inference.md) describe execution and admission.
