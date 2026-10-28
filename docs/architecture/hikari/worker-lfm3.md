# Experimental LFM3 workers

Andrew's requested BF16 evaluation is an explicit exception to the production MoE-only language design. Separate opt-in worker kinds use the existing Hikari queue, worker authentication, per-computer memory ledger and model download endpoints. They do not enter hardware tiers or the native C++ executors. Existing language, image and video workers remain available.

Each inference runs in a spawned Python executor after its worker receives a memory grant. Cancellation terminates and joins that executor before reporting cleanup. The initial experiment loads on every request and retains no CUDA model or conversation-prefix state between requests. Fixed conservative resource reservations and bounded input/context sizes make admission reviewable, but actual memory fit and speed remain unverified until the owner runs the trial.

The text checkpoint is `LiquidAI/LFM3-1.7B` at `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`, with BF16 weights and checkpoint custom code. The artifact manifest pins sizes and content hashes, including code, tokenizer, config and license. Loading uses only the verified local snapshot; it cannot resolve a mutable upstream branch or download a replacement during inference. The user must review and acknowledge the LFM Open License v1.0 through the existing download flow. Enabling a container does not acknowledge it.

Text requests name `lfm3-1.7b-bf16` explicitly. The API and worker require `HIKARI_ENABLE_LFM3_TEXT=true`. Kadan's API and runner use `KADAN_EXPERIMENTAL_LFM3_TEXT=true` to select this checkpoint for agent conversation and readiness, preserving existing session and tool orchestration. Its XML tool syntax matches the existing text harness; the harness still validates and records a command before execution. The checkpoint's own chat template is applied with thinking disabled, and any returned reasoning is removed before tool parsing. Truncated generations remain incomplete commands.

The trial limits total context to 8,192 tokens and reserves 1,024 for a reply. It rejects overflow instead of cropping history. The native model lifecycle and hardware-tier context controls do not configure this per-request executor.

See [inference expectations](../../user-expectations/hikari/inference.md) for the observable trial limits. Morning operation and rollback commands are in the experimental worker README.
