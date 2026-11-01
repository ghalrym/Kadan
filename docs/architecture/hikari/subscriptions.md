# Subscription providers

A capability can be served by the user's own subscription instead of its fixed local checkpoint. Language is wired in the current interface; this branch also wires Vision in Settings and the backend. The design keeps the provider choice per capability so other capabilities can be added without changing how subscriptions are stored. See [subscription expectations](../../user-expectations/hikari/subscriptions.md).

## Data ownership

Hikari's Postgres database owns two tables. `subscriptions` keeps one credential per provider (`claude`, `codex`): Claude's long-lived OAuth token or Codex's `auth.json`. `capability_providers` maps a capability to `local`, `claude` or `codex`; a capability without a row is local. Adding a capability means widening the `capability` check constraint, adding a field to `CapabilityProviders` and routing its endpoint; credentials are reused.

Signing out of a provider deletes its credential and resets every capability it served to `local` in the same transaction, so a request never finds a selection without a sign-in. Selecting a provider without a credential is a conflict.

## API

- `GET /v1/subscriptions` returns each provider's connection state and model, plus the capability providers.
- `DELETE /v1/subscriptions/{provider}` signs out.
- `PUT /v1/subscriptions/capabilities/{capability}` chooses the provider.
- `POST /v1/subscriptions/{provider}/login` starts a web sign-in, `GET` on the same path reports its progress and `DELETE` cancels it. `POST /v1/subscriptions/{provider}/login/code` passes Claude's sign-in code back.

Subscription writes return the full snapshot so Settings never reads its own write separately.

## Web sign-in

Users are not expected to use a terminal, so Hikari drives each CLI's own login and Settings shows only the link and code. `SubscriptionLogins` holds at most one sign-in per provider in memory; starting again replaces the earlier one, and an API restart abandons it.

- Codex runs `codex login --device-auth` with a temporary `CODEX_HOME`. Hikari reads the device link and one-time code from its output. When the process exits successfully, the `auth.json` it wrote becomes the stored credential.
- Claude runs `claude setup-token`, an interactive terminal UI, under a wide pseudo-terminal so its authorize link stays on one line. The code the Claude page shows is typed into that terminal; the long-lived OAuth token it prints becomes the stored credential. An `OAuth error` fails the sign-in immediately rather than waiting on its retry prompt.

Output is stripped of terminal escapes before parsing. Sign-ins expire after 15 minutes, matching Codex's code lifetime. Cancellation, failure and completion all kill the process and delete its temporary home.

## Language routing

`POST /v1/chat/completions` reads the language provider before queueing. With `local` it follows the existing worker path. Otherwise `SubscriptionStream` serves the turn and the queue, memory scheduler and LFM worker are never involved. The request model stays `lfm3-1.7b-bf16`, so Kadan, Hikari Chat and API clients need no change; responses report the subscription model instead.

Hikari drives each provider's own CLI rather than calling provider APIs directly, because subscription sign-ins are issued to those CLIs:

- Claude runs `claude -p` with `claude-haiku-5-5`, `CLAUDE_CODE_OAUTH_TOKEN`, all built-in tools disabled and stream JSON output, so text deltas stream through.
- Codex runs `codex exec --json` with `gpt-6.1-terra` in a read-only sandbox. Its final agent message arrives at the end of the turn.

Each run gets a fresh temporary home. Codex rotates its tokens inside `auth.json`; after a run Hikari saves the refreshed file only if the stored credential is unchanged, so a user's replacement or disconnect wins over a concurrent refresh. The temporary home is deleted afterwards.

The CLIs never see Hikari's tool schemas as native tools. The request's system messages and tool definitions become instructions in the local checkpoint's `<tool_call>` format, and history is rendered as a tagged transcript with earlier tool calls and `<tool_response>` results. Replies therefore parse through Kadan's existing tool-call parser. Output is returned in the same JSON and SSE shapes as local completions, always finishing with `stop`; `max_tokens`, `reasoning_tokens` and prefix reuse do not apply.

Client disconnects and stream closure kill and reap the CLI process. A non-zero exit, a provider error event or a missing CLI fails the turn with the provider's message; there is no fallback to local.

Kadan's readiness check reads `/v1/subscriptions` first and reports language as ready while a subscription serves it, without requiring the LFM3 download. The API image installs pinned `@anthropic-ai/claude-code` and `@openai/codex` packages.

## Vision routing on this branch

`capabilities.vision` and `PUT /v1/subscriptions/capabilities/vision` use the existing `local`, `claude`, `codex` values. UI copy can call `codex` ChatGPT without changing the stored provider identifier. Migration 0004 widens the capability constraint; existing credentials and language selections remain intact. Connection state in the subscription snapshot is the readiness signal for subscription Vision, independent of local worker/download state. It is not proof that the provider will accept a request.

The existing `/v1/vision/describe` request remains unchanged. A subscription selection bypasses the queue and local memory scheduler; `{model, text}` reports the actual provider model. `max_new_tokens` is rejected for subscriptions because these adapters cannot enforce that budget. Local worker result validation retains the fixed local model requirement.

`SubscriptionVision` reuses the CLI lifecycle, event readers and conditional Codex credential refresh from `SubscriptionStream`. Each observation has its own temporary home and process, independent of language. Disconnect and cancellation kill and reap that process before deleting its screenshot and credentials. A concurrent sign-out or replacement wins over refresh. Provider changes affect new requests; active work is neither replayed nor silently rerouted.

Claude receives an `--input-format stream-json` user message with a base64 image source and text question. Codex receives an actual PNG/JPEG file through `exec --image`; a textual filename is never used as image input. Both paths validate image encoding, format and dimensions before starting a process. The installed pinned CLI help and official [Claude streaming image documentation](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode) and [Codex CLI documentation](https://developers.openai.com/codex/cli/features) support these transports. Unit validation mocks processes; account permissions, model availability and visual quality still require owner validation. No paid inference was used. This is branch implementation, not verified master behavior.
## Image generation backend on this branch

This branch adds `image` to `CapabilityProviders` and the capability constraint, with only `local` and `codex` allowed for images. Migration 0003 widens the existing constraint without recreating credentials or rewriting language settings. Migration 0004 adds Vision while preserving the Image capability and its provider constraint. No behavior here is asserted as verified on master; UI integration and owner validation remain pending.

`PUT /v1/subscriptions/capabilities/image` accepts the existing provider request. Claude is rejected with a conflict. `GET /v1/subscriptions` includes `capabilities.image`; a selected Codex connection means generation is eligible without local readiness, not that account entitlement or remaining quota was verified. Disconnect resets every served capability in one transaction. Running requests retain their credential snapshot. Token refresh uses the existing conditional credential update so disconnect or replacement wins.

`POST /v1/images/generations` follows the image selection before native submission. The optional request `provider: local` explicitly uses Qwen; it is stripped before native serialization. Kadan pet generation supplies that override because its stages require local seeds and native reference conditioning. `/v1/images/edits` remains local for both source and reference inputs, independently of the generation provider. There is no subscription editing contract or silent provider fallback.

Subscription generation starts the pinned Codex 0.163.0 `app-server` over stdio with the stored `auth.json` in an isolated temporary home. It initializes JSON-RPC, checks `modelProvider/capabilities/read.imageGeneration`, then starts ephemeral threads and turns using the existing fixed Codex language model to invoke the built-in image tool. This capability check is not a live entitlement test. The local checkpoint queue and RAM/VRAM ledger are never entered. Count is implemented as one independent generation turn per requested image; no native cohort scheduling or capacity hint applies. Seeds and nondefault steps fail before starting the process. Aspect is included in the prompt and not enforced by cropping or resizing.

Only `item/completed` with `item.type: imageGeneration`, `status: completed`, no failure and valid PNG bytes in `result` contributes output. The turn must also complete successfully and contain exactly one artwork item. `savedPath` and assistant prose are not used to locate or substitute output. The response preserves `ImageSet`, with empty `seeds`, requested `aspect`, actual PNG dimensions in `meta` and no `next_batch_capacity`. It does not emit native progress. Multi-image failure discards partial output; provider usage already consumed may count.

Image UUID cancellation joins a subscription task's process reap and temporary-home cleanup before acknowledgment, using the same endpoint as native work. Browser disconnect and service shutdown also cancel and clean up. Successful bytes return through the existing Kadan/Tsuchi delivery pipeline. Temporary provider files are never durable result storage.

### Interface evidence and limits

The official [image generation guide](https://learn.chatgpt.com/docs/image-generation) establishes subscription usage, not an RPC contract. The installed pinned CLI's generated experimental JSON schema establishes `imageGeneration` items with `result`, `status`, optional `failure` and `savedPath`, and the provider capability read method. The official [app-server guide](https://learn.chatgpt.com/docs/app-server) establishes initialization and thread/turn lifecycle. OpenAI's public Codex image-generation extension source decodes base64 image results and publishes completed/failed image items; its current source may differ from the pinned binary. Unit fixtures exercise the installed schema shape, not real account access. No paid/live generation or inference was run. Provider availability, output quality and the assembled UI require owner validation.

The [token-sharing preview restrictions](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations) concern the separate OAuth Responses provider configuration and exclude hosted image tools there. This implementation retains Codex's own stored sign-in and built-in image tool; it does not send subscription credentials to the paid image API or claim that token-sharing route supports images.
