# Kadan mock API

Start both services from the repository root:

```sh
docker compose up --build -d
```

Start only the API with `docker compose up --build -d api`.
The API listens at http://localhost:8000; interactive documentation is at
http://localhost:8000/docs and the schema is at http://localhost:8000/openapi.json.
The existing frontend remains on port 5173 and still uses its own fixtures.

The generated frontend API contract is in `frontend/openapi.json`. It includes
the routes, request and response schemas, and validation constraints from FastAPI
and Pydantic. Regenerate it from the repository root after changing API contracts:

```sh
.venv/bin/python -m api.export_openapi > frontend/openapi.json
```

This command imports the app directly; it does not require a running server.
Treat the spec as generated output and update route models rather than editing it.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health and mock mode |
| GET | `/v1/metrics` | Sidebar resources and request dashboard statistics |
| GET | `/v1/requests` | History; `type`, `status`, `search`, `offset`, `limit` filters |
| GET | `/v1/requests/{request_id}` | Request details |
| GET | `/v1/chat/messages` | Initial conversation |
| POST | `/v1/chat/completions` | Fixed assistant response |
| GET | `/v1/decisions` | Initial state, typed questions, empty answers |
| POST | `/v1/decisions` | Validate questions; return the playground's empty answers |
| GET | `/v1/images` | Both existing image sets |
| POST | `/v1/images/generations` | Fixed generated image set |
| POST | `/v1/images/edits` | Fixed edited image set |
| GET | `/v1/videos` | Three existing video jobs |
| GET | `/v1/videos/{video_id}` | Job details and frozen progress |
| POST | `/v1/videos/generations` | Existing queued job, HTTP 202 |
| GET | `/v1/audio/speech` | Generated audio metadata and composer defaults |
| POST | `/v1/audio/speech` | Describe/clone voice inputs; fixed audio metadata |
| POST | `/v1/audio/transcriptions` | Fixed transcript from the request log |
| GET | `/v1/settings` | Model choices and Whisper formatting setting |
| PUT | `/v1/settings` | Validate selections; return unchanged settings |

All request bodies and successful responses use Pydantic models. Transport models
live in their route files; domain models live in `pydantic_models`. Route modules
follow their URL hierarchy, with collection routes in package `__init__.py` when
they have child routes. Existing camelCase fixture fields are preserved using
Pydantic aliases. Other API fields use snake_case. Invalid inputs return FastAPI's
422 validation response; missing request/video IDs return 404.

Mock data is defined directly as Python Pydantic instances in the route files that
serve it. The values match `frontend/src/data/playground.ts`,
`frontend/src/data/requests.ts`, and values embedded in the page components.
Node and the frontend are not runtime dependencies.
Generation inputs are validated but do not change outputs. No writes are persisted,
no inference runs, and jobs do not progress. Settings PUT is a validation-only mock.
Media responses contain the UI's placeholder metadata, with no real media URLs.
Image, recording, and voice sample inputs are JSON string references; files are not
uploaded or fetched in this mock. There is no authentication or live event stream.
These are frontend-oriented contracts, not OpenAI-compatible payloads.

Examples:

```sh
curl http://localhost:8000/v1/requests?type=LLM
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","text":"What caused the latency spike?"}]}'
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"script":"Hello","voice":{"mode":"describe","description":"Warm and calm"}}'
```

For local Python development, run from the repository root:

```sh
python -m venv .venv
.venv/bin/pip install -r api/requirements.txt
.venv/bin/python -m api
```

Tests under `api/tests/` mirror the API source folders and module names. For example,
`routes/v1/images/edits.py` is tested by `tests/routes/v1/images/test_edits.py`.
See the root `AGENTS.md` for folder responsibilities.

Run unit tests with `.venv/bin/python -m unittest discover -s api/tests -v`.
