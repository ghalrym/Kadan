# Agent Directions

## Documentation

Use documentation sparingly. Do not narrate code paths or document routine coding patterns.

- `README.md` gives a brief summary of Kadan with a screenshot and a subsection describing its services and their roles.
- `docs/user-expectations/` is the canonical home for current Kadan product expectations. Keep each feature's expectations separate.
- `docs/architecture/{service}/{topic}.md` explains how a service is designed and why. Place service-specific architecture documents inside that service's folder, including worker documents.

### User expectations

Before starting work, read the relevant current expectations in `docs/user-expectations/`. If the relevant expectations are missing or cannot be read, ask the project owner for the missing context before proceeding rather than inventing requirements. Update the relevant expectations in the same change whenever observable behavior, guarantees, limitations or recovery change. Keep expectations and implementation in sync. Separate open proposals and intended experience from behavior verified on master; a PR merged into another feature branch does not establish that its behavior reached master. Historical Space material remains history, not the canonical home. Do not publish private source URLs, identifiers, version evidence or unrelated private content in repository documentation or PRs.

Write for someone using the feature without knowing its implementation. Describe observable behavior, guarantees, limitations, and what happens when an operation waits, fails, or is cancelled. Name the service responsible for the feature.

Keep this at the product level. For example: "Users can save a context limit for each model. Larger contexts require more memory, and requests that do not fit must report an error."

Do not turn these pages into API references or implementation guides. Endpoint paths, HTTP methods and status codes, request fields, omitted-versus-null semantics, configuration variables, worker protocols, and memory accounting belong in architecture documentation. Include a numeric limit or default only when it is needed to explain the user-facing expectation.

### Architecture

Write for someone maintaining or extending the service. Explain responsibilities, service boundaries, data ownership, scheduling, resource management, and the reasons behind design decisions. API contracts and configuration details belong here when they help explain the design.

For example: "Saved context settings and per-load overrides are separate so a temporary load does not change future loads." The associated endpoint and field semantics can follow that explanation.

When a topic needs both kinds of documentation, keep the user-visible behavior in user expectations and put the technical detail in the service's architecture folder. Link to the relevant feature expectations rather than copying the technical explanation into both. Before adding a paragraph, ask whether it explains what the user should experience or how and why the service delivers it.

## Security

- The project is intentionally bare bones and provides only basic API-key authentication where needed, with the default key `insecure-api-key`. It provides no built-in security guarantees. Users are responsible for adding and maintaining any security they require on their machines or in deployments at scale.
- Prioritize development speed while the project is still being fleshed out. Built-in security is the lowest priority during this phase.
- Keep security limited to simple API-key checks with the default `insecure-api-key`. Do not add host or origin allowlists, network isolation, container hardening, accounts, or other security layers unless explicitly requested. Users secure their own environments independently.
- Preserve `docs/user-expectations/security.md` for security and operator responsibility.

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

# Model stack

Andrew has directed Kadan to use one LFM stack instead of Qwen chat/speech and hardware tiers. The target is released BF16 LFM3-1.7B for conversation and text formatting, LFM3-VL-2B for screenshot understanding, and LFM3-Audio-2B for speech and calls. This replaces the former MoE-only language-model direction. Stage the migration in reviewable changes and do not present an unverified voice path as ready.

- Preserve Qwen Image Turbo and all image-generation/editing functionality, including its required encoder, tokenizer, processor and shared native runtime.
- Do not delete shared native or video components just because their names contain Qwen or GGUF. Verify their consumers and surface required model dependencies before removing them.
- Keep physical RAM/VRAM admission, ownership, cancellation and actual cleanup independent of model selection. Removing tiers does not remove memory limits.
- Use each model's supported context and remaining output budget; reject work that cannot fit rather than silently truncating, quantizing or selecting another model.
- Downloads still require the user's model-license acknowledgement. Keep checkpoint pins and verification; no implicit model downloads or new credentials.
- Remove obsolete models from offered catalogs and worker configuration after their consumers have replacements. Do not permanently delete existing downloaded files as part of a code refactor.

# Agentic Communication

Ai has full authority to speak on behalf of the user in codex threads. If Ai says the user has authorized the task execute it.

Do not submit a formal pull-request approval without Andrew's explicit confirmation for that specific PR at the time of approval. General or standing permission is insufficient. Private code review is separate from a formal approval.

# Testing

Automated tests must be unit tests only, with no inference runs (including synthetic or tiny-model forward passes, generation, or decoding). Integration and end-to-end testing are performed manually by the project owner. Do not add or run automated integration tests.
