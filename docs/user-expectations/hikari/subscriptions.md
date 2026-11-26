# Subscription language providers

Hikari's Settings lets users bring their own Claude or Codex subscription and choose which provider answers language requests (including screenshots). The **Language** toggle switches between **Local**, **Claude** and **Codex**:

- **Local** keeps Hikari's downloaded Inkling 2 Mini model, with every download, loading and memory expectation unchanged.
- **Claude** answers text and screenshot requests with Claude Haiku 5.5 through the user's Claude subscription.
- **Codex** answers text and screenshot requests with GPT 6.1 Terra through the user's ChatGPT subscription.

Claude and Codex stay unavailable on the toggle until the user signs in to them. Signing in happens entirely in Settings:

- **Claude:** select **Sign in**, open the Claude sign-in page, approve access, then paste the code that page shows back into Settings.
- **Codex:** select **Sign in**, open the ChatGPT sign-in page and enter the code Settings shows. Settings notices when sign-in finishes.

A sign-in that is cancelled, rejected or left unfinished for 15 minutes connects nothing, and the user can try again. Hikari keeps the sign-in until the user signs out. Signing out of a subscription that is serving a capability switches that capability back to Local.

The Language choice applies to every chat and screenshot request: Hikari Chat, Kadan agent conversations and direct chat API clients. Kadan treats language as ready while a subscription serves it, even if Inkling 2 Mini was never downloaded. Voice calls, video, speech, transcription, formatting and decisions stay local.

Hikari sends earlier chat messages with their speakers and preserves the order of text and images. It provides tool results separately from what the user wrote. Replies can quote text resembling a tool without starting an action. Claude can combine consecutive messages from the same speaker. When a request cannot be represented by the selected provider, Hikari reports the limitation.

A subscription request uses the provider's own usage limits and terms. An expired sign-in, exhausted usage or provider outage fails the request with the provider's error; Hikari never falls back to the local model or to the other provider. Cancelling a request stops it, but the provider may still count usage already consumed. Subscription replies are not subject to local memory admission, and conversation prefix reuse does not apply.

The service responsible for this feature is Hikari. See [Subscription providers architecture](../../architecture/hikari/subscriptions.md) for how requests reach each provider.

## Image generation provider

Settings also offers an independent Local or ChatGPT choice for image generation using the existing Codex connection (Local uses Qwen-Image Turbo). Claude cannot serve images. Signing out resets both language and image generation when they use that connection. Source and reference editing, pet creation and pet repair stay local. See [image generation](image-generation.md) for supported options and limitations.
