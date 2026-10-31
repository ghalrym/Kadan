# Language-model lifecycle and context

The public lifecycle describes the fixed LFM3 worker, whose executor loads and exits per request. It does not describe a reusable native engine. See [language lifecycle expectations](../../user-expectations/hikari/model-lifecycle.md).

`GET /model-lifecycle` reports `loading` while LFM language work is outstanding and `unloaded` while the connected worker is idle with a complete checkpoint. Missing workers, incomplete checkpoints and unconfirmed cleanup report an error. The fixed context comes from the worker catalog; an idle worker has no effective resident context or independent output-token cap. A download does not establish successful inference.

`POST /model-lifecycle/unload` cancels owned language jobs, joins cleanup, and requests confirmed LFM offload before returning status. Physical charges remain until release is confirmed. It neither removes checkpoint files nor cancels media work. Public manual loading and saved-context endpoints are removed because the per-request LFM executor cannot honor them.

The retained native MoE implementation still supports explicit internal checkpoint/context requests and its existing prefix cache, admission, cancellation and cleanup rules. It has no hardware tiers or saved model selection and is not a public alternative to LFM. Its placement and execution are described in [Language worker](worker-llm.md) and [Inference architecture](inference.md).
