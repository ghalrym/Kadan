# Language-model loading and context

Hikari uses its fixed LFM conversation model. Each request loads an executor after memory admission and releases it after completion or cancellation. Download readiness and active execution are distinct: checking status never loads the model, and a downloaded checkpoint does not guarantee that a request fits. Manual preloading is unavailable in this implementation.

## Context limits

The context limit controls the combined prompt and reply budget in tokens. Settings saves the limit for the language model, or uses the checkpoint's architecture maximum. The next request uses the saved limit; changing it does not resize a request already executing. Oversized requests fail instead of silently removing history. Larger windows require more memory, and the advertised maximum is not a physical-fit guarantee.

## Releasing work

Unloading cancels queued and active conversation work and waits for executor cleanup before releasing its grant. It does not cancel other inference types or delete downloads. Later conversation work loads another executor. This implementation retains no resident language model or reusable prefix between requests.
