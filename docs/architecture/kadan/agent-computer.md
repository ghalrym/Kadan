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

## Shared viewing and recovery

The default deployment supplies one sandbox for all agents. Agent card labels do not establish separate execution environments. Kadan relays the viewer connection; Watching disables viewer input, while Take control enables it without pausing the runner or acquiring an exclusive lock. Disconnecting only closes that viewer.

The web viewer tracks its held keys through noVNC's public `sendKey` method, only while connected and controlling. It sends their releases before enabling `viewOnly` or disconnecting on close. This ordering matters in noVNC 1.7.0: setting `viewOnly` disables sending before keyboard ungrab emits releases. Public connect/disconnect events bound key ownership to that client; an unexpected disconnect discards tracking without assuming remote cleanup.

Tool results are bounded before entering the transcript, with binary results described by size. A timeout does not prove the sandbox rejected an operation. Recovery must inspect uncertain effects rather than replaying state changes.

Certain stalled-browser reads may restart the browser and retry once, reporting the reset to the agent. This can discard tabs and page state. Other uncertain browser operations require inspection. Viewer reconnection does not invoke that recovery.

See [Viewing and controlling the computer](https://chatgpt.com/space/page_bea5973327d08191afd7e289509c4874) for user expectations.
