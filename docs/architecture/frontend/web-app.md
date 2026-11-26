# Web app

`services/frontend` is the single browser app for Kadan, Tsuchi and Hikari. It
replaces the three standalone frontends so there is one container, one origin and
one dependency set. Kadan is the home page; Tsuchi and Hikari are sections that
keep their own layouts and navigation. User-facing behavior is described in
[Kadan navigation](../../user-expectations/kadan/navigation.md) and
[service status](../../user-expectations/kadan/service-status.md).

## Sections

| Path | Section | Source |
| --- | --- | --- |
| `/` and anything not below | Kadan | `src/kadan/` |
| `/tsuchi` | Tsuchi | `src/tsuchi/` |
| `/hikari/*` | Hikari | `src/hikari/` |

`src/main.tsx` owns the router and lazily imports each section's `index.tsx`, so
a section's code and stylesheet load on first visit. Links between sections use
client-side navigation (`Link` from `react-router`), so leaving Kadan unmounts it.
Kadan keeps per-chat drafts, unsent messages and the chosen persona in session
storage for that reason, and reloads its agent and chats when it remounts.

Kadan and Tsuchi still use component state rather than URLs for their internal
pages. Hikari's routes are nested under `/hikari`, and its in-app links carry that
prefix. Tsuchi's resource deep links use `/tsuchi?agent=…&resource=…`.

The sections were moved in largely unchanged to keep this first merge small. They
still have separate component libraries, API helpers and design tokens; sharing
them is follow-up work.

## API routing

The Vite server proxies each backend under the same origin:

| Browser path | Backend | Notes |
| --- | --- | --- |
| `/api` | `kadan-api` | WebSocket upgrades relay voice calls. |
| `/tsuchi/api` | `tsuchi-api` | Rewritten to `/api`; Tsuchi and Kadan both use `/api`. |
| `/v1`, `/model-lifecycle`, `/health` | `hikari-api` | Paths unchanged, so Hikari's generated client works as-is. |
| `/openapi.json`, `/docs` | `hikari-api` | Hikari's API Access page links here. |

`KADAN_API_URL`, `TSUCHI_API_URL` and `HIKARI_API_URL` choose the targets;
Compose points them at the service containers. Kadan's API no longer reports
browser URLs for Tsuchi or Hikari, since their pages share Kadan's origin.

## Stylesheets

Each section was written as a standalone app with global selectors (`body`,
`:root`, `.sidebar`, `.brand`, …) that collide with each other.
`config/scopeSectionCss.ts` is a PostCSS plugin that prefixes every rule in
`src/<section>/` with `html[data-app="<section>"]`, and the shell sets `data-app`
on `<html>` for the open section (`index.html` sets it before the app loads to
avoid an unstyled flash). Rules for the document root (`html`, `:root`,
Tsuchi's `[data-theme]` and root pseudo-elements such as `::view-transition-*`)
attach to `html` itself instead of becoming descendants. Keyframes and font faces
stay global; the duplicate keyframe names across sections are identical.

## Tests and tooling

Unit tests stay beside each section (`src/<section>/**/*.test.mjs`) and run with
one `npm test`. They transpile sources with TypeScript and mock imports by name,
so a new import in a tested module needs a matching mock. TypeScript is pinned to
5.9 because TypeScript 6 changes the module interop of that transpile step.
Hikari's OpenAPI schema (`openapi.json`) and client generator config live at the
app root; `services/hikari/bin/hikari.py generate api` writes the client into
`src/hikari/api/generated`.
