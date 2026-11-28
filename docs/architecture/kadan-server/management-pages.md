# Management pages

The [Kadan Server expectations](../../user-expectations/kadan-server/overview.md) describe what people see. This note explains where the pages live and which services they read.

## One web route, embedded by the desktop window

The pages are a section of the main web app (`services/frontend/src/server/`), routed at `/server/*` beside the Kadan client. They replaced the separate Hikari and Tsuchi web apps, whose routes now redirect (`/hikari/*` to `/server/models`, `/tsuchi/*` to `/server/agent`).

Serving them from the combined stack rather than from the Tauri bundle has three effects:

- **Same origin.** The pages call `/kadan/api`, `/tsuchi/api` and `/hikari/v1` directly. The combined API has no CORS setup, and the Tauri webview's origin would otherwise need one.
- **Updates ship with the services.** Refreshing `services/` and `services/frontend/dist` on a server updates the pages without rebuilding the Rust `kadan-server` binary.
- **Browsers work too.** Anyone who can reach the server can open `/server`, which keeps every management flow available in the web UI.

The Tauri window (`apps/kadan-server/src/App.tsx`) shows `${localHostUrl}/server` in an iframe on its **Kadan Server** tab once the `api` service is running. Things that need the supervisor's Tauri commands stay native: start and stop, **This computer** (config, service restarts, hardware, updates) and **Logs**. The pages hide **Open Kadan** when `window.self !== window.top`, since navigating the iframe to the client would trap it.

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
