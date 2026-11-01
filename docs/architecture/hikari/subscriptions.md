# Subscription providers

A capability can be served by the user's own subscription instead of its fixed local checkpoint. Only language is wired today. The design keeps the provider choice per capability so other capabilities can be added without changing how subscriptions are stored. See [subscription expectations](../../user-expectations/hikari/subscriptions.md).

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
