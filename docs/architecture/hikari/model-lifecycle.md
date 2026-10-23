# Language-model lifecycle and context

Hikari separates a model's saved settings from its current loaded state. This lets clients save a context preference without loading the model, or choose a different context for one load without replacing that preference. See the privately provided expectations for language-model loading and context.

## Loading and status

These endpoints belong to Hikari's API:

| Endpoint | Purpose |
| --- | --- |
| `GET /model-lifecycle` | Read model state and context limits without loading a model or starting inference. |
| `POST /model-lifecycle/load` | Load the selected tier's language model, or another supported model supplied through `model_id`. |
| `POST /model-lifecycle/unload` | Cancel pending language work and wait for cleanup. |

Explicit loading uses the same queue, preparation, and memory checks as inference. Keeping these operations together means a manual load cannot bypass memory admission or compete with inference outside the scheduler. Loading can wait, download missing checkpoints, or fail. Hikari must not silently substitute another model to satisfy the request.

The load endpoint returns HTTP 202, but that response does not guarantee that future prompts will fit in memory. Clients should read the reported model state rather than infer readiness from the response code or the selected tier.

Status reports `unloaded`, `loading`, `ready`, `offloaded`, `unloading`, or `error`. It also distinguishes the configured context limit, the effective limit, and the checkpoint's supported maximum. These values are separate because a saved preference does not prove that a model has loaded with that setting.

Readiness follows the connected workers' observed resident models. A budget grant alone does not establish residency: resizing can close an engine while leaving its worker permitted to load again. Confirmed release or loss of the last matching worker reports `unloaded` and no effective loaded context, while preserving the configured context observation. Another resident copy can still establish readiness using its own effective context. A settled or disconnected load cannot leave status stuck at `loading`. Starting another inference type does not itself establish that the language model was offloaded. This reporting does not change memory scheduling or retained conversation state.

A failed or cancelled language request makes that worker's residency observation unknown: preparation cancellation may retain an engine, while execution failure may destroy it. Unknown observations cannot establish readiness or an effective context. When no confirmed resident copy remains, reporting uses the existing error state. A successful request or an authoritative worker snapshot resolves the residency uncertainty. A snapshot can confirm a resident model without recovering its effective context; that limit remains absent until observed again.

## Saved context and load overrides

Each language checkpoint defaults to 65,536 context tokens. `PUT /v1/models/{model_id}/context` saves its `context_limit`. A positive integer sets the token limit; `null` selects the checkpoint's maximum supported context on load. Boolean values are rejected.

For `POST /model-lifecycle/load`, the meaning of `context_limit` depends on whether it was supplied:

| Value | Effect for this load |
| --- | --- |
| Omitted | Use the saved setting, or the default if none is saved. |
| `null` | Use the checkpoint's maximum supported context. |
| Positive integer | Use this token limit. |

Load overrides do not change the saved setting. Saving a setting does not immediately resize an already loaded model. This separation keeps a temporary load choice from changing later loads unexpectedly.

Hikari rejects limits above the architecture maximum when that maximum is known. Loading also validates the limit against the actual checkpoint. A supported context length can still exceed available physical memory, so context validation does not replace memory admission.

Context is measured in tokens, not characters. It is separate from the current reply cap of 256 output tokens. Requests that exceed the admitted context must fail explicitly rather than silently truncate conversation history.

## Unloading and cleanup

Explicit unload requests cancellation of pending language work and waits for that work's cleanup before resources can be reused. It does not cancel other inference types or delete downloaded checkpoints. Later inference can load the model again.

Automatic model expiry and retention of reusable conversation state have separate lifetimes. See [Inference architecture](inference.md) for memory reclamation and conversation state retention, and [Language worker](worker-llm.md) for model placement.
