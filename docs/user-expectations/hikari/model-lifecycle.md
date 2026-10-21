# Language-model loading and context

Hikari manages language-model loading automatically. Users can also load a model before starting a conversation or unload it to free memory.

## Loading and status

Hikari uses the selected language model and must not silently substitute another model to make a request fit. Loading can take time, require a download, or fail if the model cannot run on the available hardware.

Users should be able to check whether the model is loading, ready, unloaded, or has failed. Checking status does not load a model. Selecting a model does not mean it is ready to use.

## Context limits

The context limit controls how much text a model can work with at once, measured in tokens. It is separate from the maximum reply length.

Users can save a context limit for each model or choose to use the model's maximum supported context. A temporary choice for one load does not replace the saved setting. Changing the saved setting does not immediately resize a model that is already loaded.

Larger contexts need more memory. A model's supported maximum may exceed what the user's machine can handle. Hikari must reject unsupported limits and report requests that do not fit, rather than silently removing conversation history.

## Unloading

Unloading a language model cancels pending language work and waits for cleanup before releasing its resources. It does not cancel other inference types or delete downloaded models. Later inference can load the model again.

See [Inference operations](inference.md) for automatic unloading and conversation prefix reuse.
