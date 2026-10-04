# Kadan frontend

Static React + TypeScript implementation of `../design-work/Kadan Dashboard.html`, built with Vite and React Router.

## Development

From the repository root, start the frontend in Docker:

```sh
docker compose up -d frontend
```

Open `http://localhost:5173`. The container mounts `frontend/` directly, and Vite
polls for file changes to provide hot reloading. Container dependencies live in a
separate Docker volume, so they do not overwrite your local `node_modules`.
The container runs `npm ci` at startup using the committed lockfile.

```sh
docker compose logs -f frontend   # Follow startup and dev-server logs
docker compose restart frontend # Reinstall after package/lockfile changes
docker compose down             # Stop the services
```

Stop any locally running Vite server before starting Docker; both use port 5173.
Compose also provides the separate mock API on port 8000.

To run directly on your machine instead:

```sh
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`).

```sh
npm run build   # TypeScript check and production build
npm run lint    # Oxlint
npm run preview
```

## Structure

- `src/App.tsx` defines the routes and shared application layout.
- `src/pages/` contains one component for each main screen.
- `src/components/` contains the sidebar/layout, shared form controls, media views, question cards, video cards, and request drawer.
- `src/data/` holds navigation metadata and fixed design fixtures.
- `src/styles/` contains local font declarations, shared tokens and base rules, layout, reusable component styles, and page-specific styles. All styling lives in stylesheets; components have no inline CSS.
- `src/assets/fonts/` and `public/kadan.svg` preserve the fonts and logo bundled in the supplied design.

## Routes

| URL                    | Screen                                     |
| ---------------------- | ------------------------------------------ |
| `/requests`            | Request metrics and history                |
| `/requests/:requestId` | Request detail drawer over the history     |
| `/chat`                | Chat                                       |
| `/decisions`           | State, question configuration, and answers |
| `/image`               | Image generation                           |
| `/image/edit`          | Image editing                              |
| `/video`               | Video generation and queue                 |
| `/tts`                 | Described voice synthesis                  |
| `/tts/clone`           | Voice cloning                              |
| `/stt`                 | Speech recording and transcription         |
| `/api`                 | API documentation placeholder              |
| `/settings`            | Model settings                             |

The root redirects to `/requests`. Unknown URLs and unknown request IDs have explicit not-found views. A production host must serve `index.html` for frontend routes so direct links and refreshes work (Vite already does this during development).

## Static scope

Navigation, request-detail links, and the Advanced disclosure work. Everything else is a visual placeholder: inputs are read-only, action buttons and settings are disabled, and no server requests, timers, storage, recording, uploads, downloads, or simulated generation run. “Online”, “Live”, metrics, request output, and queue progress are frozen design fixtures, not server status. Image tiles intentionally keep the reference's striped placeholders.

The Python server remains separate. When the backend is ready, replace the fixtures with API data and enable the appropriate controls.

## Generated API client

[`openapi.json`](openapi.json) is generated from the FastAPI routes and Pydantic
models. Hey API generates typed Fetch functions and request/response types in
`src/api/generated/`, configured by `openapi-ts.config.ts`. The pages still use
their existing mock data; generating the client does not make any API calls.

Regenerate it from the repository root with the API's Python dependencies installed:

```sh
.venv/bin/python -m api.export_openapi > frontend/openapi.json
npm --prefix frontend run generate:api
```

The second command can run independently when the spec is already current, including
inside the frontend container: `docker compose exec frontend npm run generate:api`.
Edit the API routes and models, then regenerate; do not edit the spec or generated
TypeScript by hand. Route `operation_id` values determine the SDK function names.
Hey API is pinned for reproducible generation. The `js-yaml` override keeps its
schema parser on the patched 4.3.2 release until the upstream dependency is updated.
The running API also serves `/openapi.json` and interactive documentation at
`http://localhost:8000/docs`.

For future frontend integration, import functions and types directly:

```ts
import { createCompletion, type CompletionRequest } from './api/generated'

const body: CompletionRequest = {
  messages: [{ role: 'user', text: 'Hello' }],
}
const { data, error } = await createCompletion({ body })
```

The generated client uses same-origin URLs. Vite proxies `/v1` and `/health` to
`http://127.0.0.1:8000` locally, or `http://api:8000` in Compose via `API_PROXY_TARGET`.
Start both services with `docker compose up --build -d`. Production hosting must
route these paths to the API, or configure a different API origin using the
generated client's `setConfig({ baseUrl })` method and allow that origin on the API.
