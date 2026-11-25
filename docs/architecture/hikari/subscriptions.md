# Subscription providers

A capability can be served by the user's own subscription instead of its local checkpoint. Language (including multimodal screenshot vision) and Image generation are wired in Settings and the backend. The design keeps the provider choice per capability so other capabilities can be added without changing how subscriptions are stored. See [subscription expectations](../../user-expectations/hikari/subscriptions.md).

## Data ownership

Hikari's Postgres database owns two tables. `subscriptions` keeps one credential per provider (`claude`, `codex`): Claude's long-lived OAuth token or Codex's `auth.json`. `capability_providers` maps a capability (`language`, `image`) to `local`, `claude` or `codex`; a capability without a row is local. Migration 0005 drops the retired `vision` capability row and narrows the check constraint back to `language` and `image`, since multimodal images now flow through `POST /v1/chat/completions`. Adding a capability means widening the `capability` check constraint, adding a field to `CapabilityProviders` and routing its endpoint; credentials are reused.

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

## Language routing (including multimodal vision)

`POST /v1/chat/completions` reads the language provider before queueing. Local requests use the native worker. Subscription chat and screenshot requests use the selected provider’s CLI and report its configured model. Hikari passes messages and tools as structured provider records. It keeps text and image blocks in their original order.

- **Claude** uses Claude Code 2.1.296 with built-in tools disabled. System instructions use the system prompt. Earlier turns go into a linked transcript in the request’s temporary home. Claude resumes at the record immediately before the current input. The snapshot includes the final record so recovery sees paired tool calls and results before applying that boundary. The final user input or native tool result arrives through stream JSON. Hikari registers the requested schemas with the SDK-hosted MCP protocol and returns the model’s completed tool call to the caller for execution. The stream preserves text deltas and checks the terminal stop reason before publishing tool arguments. Claude accepts system instructions at the start of the request and requires a final user input or tool result for this path. Its model can combine adjacent turns with the same role. [Claude messages](https://platform.claude.com/docs/en/api/typescript/messages/create)
- **Codex** uses the pinned 0.163.0 app-server and its signed-in temporary home. It receives leading instructions via `baseInstructions`, callable schemas via `dynamicTools`, and the turn history via `thread/inject_items`. History uses Responses items for messages, calls and paired results. Starting the idle thread with empty input allows it to sample from that exact history, including when the final item is a tool result. Hikari disables environment context, shell, web search, image generation and agent delegation for these requests. The CLI can still supply model-specific internal utilities such as its clock. Hikari returns a declared dynamic call to Kadan when Codex requests execution. For a text response, it waits for the terminal turn and returns the final agent message. [Pinned turn protocol](https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server/src/request_processors/turn_processor.rs)

Kadan owns requested tool execution and persists the result. Provider text remains ordinary text even when it quotes tool markup. A malformed or interrupted provider response fails the request. Subscription generation uses the provider’s own token limits. Native reasoning budgets and prefix reuse apply only to Local.

Disconnect or completion releases the provider process and its temporary home. When Codex rotates its tokens in `auth.json`, Hikari saves the refreshed credential only if the stored credential still matches the one used to start the request. A concurrent sign-out or replacement sign-in wins. Provider failures are reported to the caller.

Kadan's readiness check reads `/v1/subscriptions` first and reports language as ready while a subscription serves it, without requiring the Inkling 2 Mini download. The API image installs pinned `@anthropic-ai/claude-code` and `@openai/codex` packages.

## Image generation backend

`CapabilityProviders` includes `image`, with only `local` and `codex` allowed for images. Migration 0003 widens the existing constraint without recreating credentials or rewriting language settings, and Migration 0005 preserves `image` while dropping the retired `vision` capability.

`PUT /v1/subscriptions/capabilities/image` accepts the existing provider request. Claude is rejected with a conflict. `GET /v1/subscriptions` includes `capabilities.image`; a selected Codex connection means generation is eligible without local readiness, not that account entitlement or remaining quota was verified. Disconnect resets every served capability in one transaction. Running requests retain their credential snapshot. Token refresh uses the existing conditional credential update so disconnect or replacement wins.

`POST /v1/images/generations` follows the image selection before native submission. The optional request `provider: local` explicitly uses Qwen; it is stripped before native serialization. Kadan pet generation supplies that override because its stages require local seeds and native reference conditioning. `/v1/images/edits` remains local for both source and reference inputs, independently of the generation provider. There is no subscription editing contract or silent provider fallback.

Subscription generation starts the pinned Codex 0.163.0 `app-server` over stdio with the stored `auth.json` in an isolated temporary home. It initializes JSON-RPC, checks `modelProvider/capabilities/read.imageGeneration`, then starts ephemeral threads and turns using the existing fixed Codex language model to invoke the built-in image tool. This capability check is not a live entitlement test. The local checkpoint queue and RAM/VRAM ledger are never entered. Count is implemented as one independent generation turn per requested image; no native cohort scheduling or capacity hint applies. Seeds and nondefault steps fail before starting the process. Aspect is included in the prompt and not enforced by cropping or resizing.

Only `item/completed` with `item.type: imageGeneration`, `status: completed`, no failure and valid PNG bytes in `result` contributes output. The turn must also complete successfully and contain exactly one artwork item. `savedPath` and assistant prose are not used to locate or substitute output. The response preserves `ImageSet`, with empty `seeds`, requested `aspect`, actual PNG dimensions in `meta` and no `next_batch_capacity`. It does not emit native progress. Multi-image failure discards partial output; provider usage already consumed may count.

Image UUID cancellation joins a subscription task's process reap and temporary-home cleanup before acknowledgment, using the same endpoint as native work. Browser disconnect and service shutdown also cancel and clean up. Successful bytes return through the existing Kadan/Tsuchi delivery pipeline. Temporary provider files are never durable result storage.

### Interface evidence and limits

The official [image generation guide](https://learn.chatgpt.com/docs/image-generation) establishes subscription usage, not an RPC contract. The installed pinned CLI's generated experimental JSON schema establishes `imageGeneration` items with `result`, `status`, optional `failure` and `savedPath`, and the provider capability read method. The official [app-server guide](https://learn.chatgpt.com/docs/app-server) establishes initialization and thread/turn lifecycle. OpenAI's public Codex image-generation extension source decodes base64 image results and publishes completed/failed image items; its current source may differ from the pinned binary. Unit fixtures exercise the installed schema shape, not real account access. No paid/live generation or inference was run. Provider availability, output quality and the assembled UI require owner validation.

The [token-sharing preview restrictions](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations) concern the separate OAuth Responses provider configuration and exclude hosted image tools there. This implementation retains Codex's own stored sign-in and built-in image tool; it does not send subscription credentials to the paid image API or claim that token-sharing route supports images.
