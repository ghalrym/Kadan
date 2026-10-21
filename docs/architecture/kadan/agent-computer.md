# Agent computer (`kadan-sandbox`)

Gives agents a real computer — shell, files, code and a visible browser — on the
shared Compose network, and lets the owner watch or take over that computer from Kadan.

## Components

- `kadan-sandbox`: the pinned AIO Sandbox image (`ghcr.io/agent-infra/sandbox`).
  It owns execution. Kadan installs nothing into it and keeps no copy of its API.
- `kadan/api/sandbox_client.py`: the only client of the sandbox's HTTP API.
- `kadan/runner/tools/`: one hand-written LangChain tool per sandbox route, with
  a Pydantic argument model. Each module's `TOOLS` list selects the enabled tools.
  `kadan-runner` binds those tools to the model when a computer is configured and
  executes the calls.
- `kadan-api` reports status and relays the screenshot and VNC stream for the
  agent card and computer viewer in the web app.

## Depends on

- The sandbox's HTTP API and its `SANDBOX_API_KEY` authentication.
- The sandbox's websockify endpoint for VNC.

## Constraints

- The sandbox shares the default Compose network with the other services.
- Sandbox API-key authentication defaults to `insecure-api-key` and can be overridden.
- The API key stays server-side. Browsers reach the computer only through Kadan.
- Tool definitions and the enabled `TOOLS` lists stay separate; only enabled
  tools are bound to the model.
- Results entering the agent transcript stay bounded, and sandbox failures are tool
  results, not run failures.
- The sandbox has no idempotency keys. A command saved before a restart is never
  re-executed unless it is a GET.
