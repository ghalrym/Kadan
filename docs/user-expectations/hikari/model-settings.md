# Model settings and downloads

A new installation has no selected hardware tier. Settings highlights a suggested tier, but that highlight does not select it or start downloads. Choose a tier for the available GPU and system RAM, review its model license notices, then confirm the download. Choosing No leaves that confirmation without starting it.

The selected tier is saved and determines its modality checkpoints and video defaults. Selecting it queues incomplete model downloads; accepting the request does not mean those downloads are finished or that a model is resident in GPU memory. Models load when inference needs them. A tier's advertised hardware sizes are starting configurations, not a guarantee that every request will fit or run at a particular speed.

Use Settings to download and inspect checkpoints before inference. In the current implementation, native request preparation can also acquire a missing checkpoint, so a direct inference request is not guaranteed to be free of downloads. An unusable existing checkpoint directory or failed acquisition can instead produce an error. Kadan's chat interface separately gates normal sending on the reported downloaded language-model state.

The progress dialog shows each assigned checkpoint separately. Queued downloads can have no observed byte total yet, and the aggregate percentage can use catalog estimates until sizes are known. Completion is established by the checkpoint state, not a rounded percentage. Failures remain visible. Resume downloads retries the selected tier's incomplete downloads; it does not require re-downloading completed checkpoints.

Continue in background, closing the dialog or leaving Settings does not cancel a download. Tier selection is disabled in this interface while the selected tier is downloading. Reopen View downloads to inspect progress. When a settings request fails or times out, refresh its state before assuming nothing changed: saving the selection and completing all downloads are separate outcomes.

API clients can queue an individual checkpoint download and request cancellation through its /v1/models/{model_id}/download endpoint. HTTP 202 acknowledges that operation, not a completed download or immediate resource release. Poll model status for the outcome. A second active download for the same model, or cancellation when none is active, produces a conflict rather than inventing a successful result. Cancellation does not delete a previously completed checkpoint.

Changing tiers can unload the previous tier's resident language model after earlier chat work finishes. It does not move or delete downloaded checkpoints or save a Hikari playground conversation. A later request can incur loading time. See [Inference operations](inference.md) for memory admission, priority, model expiry and prefix-reuse limits.

For API clients, omitted settings fields retain their values. Sending a null tier does not clear the selection. Transcription formatting has its own saved switch and per-request override; changing the tier alone does not toggle it. Settings reads and model status checks do not run inference. Preserve the model store as described in [Checkpoint storage](../../../services/hikari/README.md#checkpoint-storage).
