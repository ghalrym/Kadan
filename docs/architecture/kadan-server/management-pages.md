# Management pages

The [Kadan Server expectations](../../user-expectations/kadan-server/overview.md) describe what people see. This note explains where the pages live and which services they read.

## One web route, embedded by the desktop window

The pages are a section of the main web app (`services/frontend/src/server/`), routed at `/server/*` beside the Kadan client. They replaced the separate Hikari and Tsuchi web apps, whose routes now redirect (`/hikari/*` to `/server/models`, `/tsuchi/*` to `/server/agent`).

Serving them from the combined stack rather than from the Tauri bundle has three effects:

- **Same origin.** The pages call `/kadan/api`, `/tsuchi/api` and `/hikari/v1` directly. The combined API has no CORS setup, and the Tauri webview's origin would otherwise need one.
- **Updates ship with the web app.** Rebuilding `services/frontend/dist` updates the pages without rebuilding the Rust app.
- **Browsers work too.** Anyone who can reach the server can open `/server`, which keeps every management flow available in the web UI.

In the Kadan app the pages are served by its loopback proxy like the rest of the web app. Hosting controls that need the app itself (start, stop, service restarts, settings, logs) come from the proxy's `/__kadan_desktop__/host` endpoints and appear on Settings only when `window.__KADAN_DESKTOP__` is set; see [Desktop application](../desktop-apps.md). Because the page's own origin is the private proxy there, `address.ts` builds the address to share from the hosting computer's network address and port, or the connected computer's address.

## Styling

The section imports `src/kadan/styles.css` and sets `html[data-app="kadan"]`, so it shares the client's tokens, buttons, dialogs and sidebar. `server.css` adds `srv-*` classes, each prefixed with `:root .srv-app` to outrank the client's `html[data-app="kadan"]`-scoped element rules without editing them.

## Data sources

| Page | Reads | Writes |
| --- | --- | --- |
| Home | `GET /kadan/api/status`, `/hikari/v1/models`, `/hikari/v1/metrics`, Tsuchi `/agents/{key}/stats` | — |
| AI models | Hikari's generated client via `src/hikari/api/settings.ts` and `subscriptions.ts` | downloads with license acknowledgement, `llm_model` switches, provider choices, sign-in flows |
| Your agent | Kadan `/agents`, `/agents/{key}/pet`, `/memories`, `/persona`, `/conversations`, `/resources/{id}`; Tsuchi `/resources/{id}/versions`, trashed `/resources` and `/assets` | memory `PUT` through Kadan (so its version checks apply), forgetting via Tsuchi `DELETE /resources/{id}`, persona `PUT` through Kadan, version restore and trash restore through Tsuchi |
| Activity | `GET /hikari/v1/requests` (paged, filtered by type) | — |

The first agent the API returns is the agent, as in the client. Ability names and states are derived in `abilities.ts` from Hikari's checkpoint statuses, so the plain wording lives in one place. Models poll every 1.5 seconds while a download runs and every 10 seconds otherwise.

Hikari's playgrounds (chat, decisions, images, text to speech, speech to text) and its API-access page were removed with their UI code; the corresponding APIs remain for Kadan and direct clients. Tsuchi's namespace switcher, resource editors, search, copies, image library and activity pages were removed as well; pages, slides, folders and files are edited in Kadan's Space.
