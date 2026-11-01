# Subscription language providers

Hikari's Settings lets users bring their own Claude or Codex subscription and choose which provider answers language requests. The **Language** toggle switches between **Local**, **Claude** and **Codex**:

- **Local** keeps Hikari's downloaded LFM3 1.7B model, with every download, loading and memory expectation unchanged.
- **Claude** answers with Claude Haiku 5.5 through the user's Claude subscription.
- **Codex** answers with GPT 6.1 Terra through the user's ChatGPT subscription.

Claude and Codex stay unavailable on the toggle until the user signs in to them. Signing in happens entirely in Settings:

- **Claude:** select **Sign in**, open the Claude sign-in page, approve access, then paste the code that page shows back into Settings.
- **Codex:** select **Sign in**, open the ChatGPT sign-in page and enter the code Settings shows. Settings notices when sign-in finishes.

A sign-in that is cancelled, rejected or left unfinished for 15 minutes connects nothing, and the user can try again. Hikari keeps the sign-in until the user signs out. Signing out of a subscription that is serving language switches language back to Local.

The choice applies to every language request: Hikari Chat, Kadan agent conversations and direct chat API clients. Kadan treats language as ready while a subscription serves it, even if LFM3 was never downloaded. Voice calls, images, video, speech, transcription, formatting and decisions stay local. Vision subscription support on this branch is described below.

A subscription request uses the provider's own usage limits and terms. An expired sign-in, exhausted usage or provider outage fails the request with the provider's error; Hikari never falls back to the local model or to the other provider. Cancelling a request stops it, but the provider may still count usage already consumed. Subscription replies are not subject to local memory admission, and conversation prefix reuse does not apply.

The service responsible for this feature is Hikari. See [Subscription providers architecture](../../architecture/hikari/subscriptions.md) for how requests reach each provider.

## Vision subscriptions on this branch

Hikari also supports an independent Vision choice of Local, Claude or ChatGPT, reusing the same connected subscriptions. ChatGPT uses the existing Codex connection. The Vision selector is available in Settings on this branch; this behavior has not been verified on master.

Subscription Vision answers the existing screenshot question without downloading or loading LFM3-VL. It leaves Language unchanged. Local retains its license acknowledgement, download, loading and memory requirements. Signing out resets Vision and any other capability using that subscription to Local; it does not replay an active request. Changes apply to subsequent requests.

Screenshots retain the PNG/JPEG, size and dimension limits of Local. Explicit reply token budgets are supported only by Local; subscription requests with a budget fail with an explanation. An unavailable installation, expired sign-in, usage limit or provider error fails without another provider taking over. Cancellation stops the provider process before request cleanup completes; already consumed usage may still count. Subscription observations use no local model memory admission.
