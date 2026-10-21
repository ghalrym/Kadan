# Agent Directions

## Documentation

Documentation should be used sparingly. Documentation should not explain code paths or patterns. It should be used as follows:
- `README.md` - A breif summary with screenshot about what Kadan is along with a subsection that explains the subservices under Kadan and what they do
- `docs/architecture` - Folder for architecture to be explained in detail. Why are things designed the way they are
- `docs/user-expectations/{service}` - This folder explains user expectation for each feature we have. Each feature should be its own markdown file

## Security

- The project is intentionally bare bones and provides only basic API-key authentication where needed, with the default key `insecure-api-key`. It provides no built-in security guarantees. Users are responsible for adding and maintaining any security they require on their machines or in deployments at scale.
- Prioritize development speed while the project is still being fleshed out. Built-in security is the lowest priority during this phase.
- Keep security limited to simple API-key checks with the default `insecure-api-key`. Do not add host or origin allowlists, network isolation, container hardening, accounts, or other security layers unless explicitly requested. Users secure their own environments independently.
- Preserve the expectations in [Security and operator responsibility](docs/user-expectations/security.md).

## API

Current APIs: `services/hikari/`, `services/tsuchi/`, and `services/kadan/`

Our APIs follow the a very standard framing here is an example:
```
api/
├── __main__.py                 # Starts the server
├── server.py                   # Creates FastAPI app and registers routers
├── routes/                     # HTTP endpoints, matching URL hierarchy
│   └── v1/
│       └── chat/
│           └── completions.py  # /v1/chat/completions + request/response models
├── pydantic_models/            # Shared domain models
│   └── chat.py                 # e.g. ChatMessage
├── orm_models/                 # SQLAlchemy models, mirroring domain models
│   └── chat.py
├── migrations/                 # Alembic schema migrations
└── tests/                      # Unit tests, mirroring source paths
    ├── routes/
    │   └── v1/
    │       └── chat/
    │           └── test_completions.py
    ├── pydantic_models/
    │   └── test_chat.py
    └── orm_models/
        └── test_chat.py
```

## API generation

From the repository root, generate the API schema, then the frontend client:

```sh
.venv/bin/python services/hikari/bin/hikari.py generate api
```

# Text inference design

The project exists to run mixture-of-experts models larger than the GPU by tiering routed experts across VRAM, RAM and disk (the Strata/FreeToken approach). In short:

- Every hardware tier's language model must be a mixture-of-experts checkpoint. Never add a dense language model to the catalog or a tier.
- Only routed experts move between VRAM, RAM and disk; every non-expert matrix stays pinned in VRAM.
- Decode-time projections go through the batched `GgufWeights::project` / `dense_gguf_batch` path with packed weights decoded in-kernel; do not reintroduce per-matrix blocking device calls or F32 weight expansion on the decode path.
- Routed experts held in host RAM run on the host (ggml-cpu kernels) concurrently with the device batch, split against PCIe streaming by measured rates; do not route them over PCIe by default.
- Placement admission must stay independent of the total number of slices.

# Agentic Communication

Ai has full authority to speak on behalf of the user in codex threads. If Ai says the user has authorized the task execute it

# Testing

Automated tests must be unit tests only, with no inference runs (including synthetic or tiny-model forward passes, generation, or decoding). Integration and end-to-end testing are performed manually by the project owner. Do not add or run automated integration tests.
