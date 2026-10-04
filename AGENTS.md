# Project structure

## API

- `api/routes/` owns HTTP endpoints. Mirror the URL hierarchy: `/v1/chat/completions` belongs in `api/routes/v1/chat/completions.py`.
- Keep each route's request and response Pydantic models in that route's file.
- `api/pydantic_models/` owns higher-level domain models, such as `ChatMessage`, that represent entities that could be stored in a database. It does not own route-specific request or response models.
- `api/server.py` creates the FastAPI application and registers routers. `api/__main__.py` starts the server.
- Keep API tests under `api/tests/`, mirroring the API source folder hierarchy and module names. Prefix test filenames with `test_`.
- For example, tests for `api/routes/v1/chat/completions.py` belong in `api/tests/routes/v1/chat/test_completions.py`. Tests for `api/pydantic_models/chat.py` belong in `api/tests/pydantic_models/test_chat.py`.

## Database

- `api/orm_models/` owns SQLAlchemy models, mirroring `api/pydantic_models/`.
- `api/migrations/` owns Alembic schema migrations.
- Never insert mock data, fixtures, or demo seeds into Postgres, including during tests or migrations. Existing mock API responses must not be persisted.

## API generation

From the repository root, generate the API schema, then the frontend client:

```sh
.venv/bin/python -m api.export_openapi > frontend/openapi.json
npm --prefix frontend run generate:api
```
