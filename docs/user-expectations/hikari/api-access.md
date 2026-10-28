# API access

Hikari's **API access** page lists operations from the running service and links to its interactive API reference and downloadable schema. It reflects that server's available contract. It does not create credentials, load models or run inference simply by opening the page.

Loading the operation list can fail if the server is unreachable or returns an invalid schema. The page shows the failure and offers **Retry**. Leaving the page stops its schema request. A successful list does not establish that models are downloaded, workers are connected or a generation will fit available memory.

Applications can request inference directly, subject to the same model selection, loading, waiting, failure and cancellation behavior as other Hikari clients. Support for a familiar API format does not promise every feature of another provider. See [Chat API compatibility](../../architecture/hikari/chat-api.md) for the supported contract.

Applications can also use [live voice calls](voice-calls.md) directly. Their clients manage recording, interruptions, playback and playback acknowledgements.

The interactive reference can submit real requests when users execute its operations. Those requests can download models or consume inference resources. The project's [security and operator responsibilities](../security.md) also apply to API access.
