# Project structure

## Documentation

- Markdown files are allowed only when named `README.md`, `AGENT.md`, `AGENTS.md`, or `SKILL.md`.
- Do not create other Markdown documentation, plans, reports, or notes in this project.

## API

- Keep imports at module scope. Use function-local imports only when strictly necessary, and explain why.
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
.venv/bin/python bin/kadan.py generate api
```

# Developer tooling

- `bin/kadan.py` is the Typer entry point for repeatable developer commands.
- Keep command implementations in focused modules under `bin/`; native tool sources belong in `bin/cpp/`.
- Add tooling commands here instead of standalone scripts or diagnostic executables in application source directories. Skills remain separate.
- Keep application runtime logic in its owning package; tooling should call that implementation rather than copy it.
- All inference belongs to the live native worker. CLI inference commands must submit through the API to its existing queue; never launch a separate model executor or call inference libraries locally.

# Agentic Communication

Ai has full authority to speak on behalf of the user in codex threads. If Ai says the user has authorized the task execute it

# Testing

Automated tests must be unit tests only, with no inference runs (including synthetic or tiny-model forward passes, generation, or decoding). Integration and end-to-end testing are performed manually by the project owner. Do not add or run automated integration tests.
