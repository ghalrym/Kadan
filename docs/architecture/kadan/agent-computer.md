# Agent computer (`kadan-sandbox`)

Gives agents a real computer — shell, files, code and a visible browser — on the
shared Compose network, and lets the owner watch or take over that computer from Kadan.

## Components

- `kadan-sandbox`: the pinned AIO Sandbox image (`ghcr.io/agent-infra/sandbox`).
  It owns execution. The image also includes the pinned Codex CLI and Kadan's
  [coding supervisor](coding.md); Kadan keeps no copy of the sandbox's API.
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
  Coding submissions additionally have supervisor-owned request IDs, but are still
  inspected explicitly after an interrupted conversation action.

## Shared viewing and recovery

The default deployment supplies one sandbox for all agents. Agent card labels do not establish separate execution environments. Kadan relays the viewer connection; Watching disables viewer input, while Take control enables it without pausing the runner or acquiring an exclusive lock. Disconnecting only closes that viewer.

The web viewer tracks its held keys through noVNC's public `sendKey` method, only while connected and controlling. It sends their releases before enabling `viewOnly` or disconnecting on close. This ordering matters in noVNC 1.7.0: setting `viewOnly` disables sending before keyboard ungrab emits releases. Public connect/disconnect events bound key ownership to that client; an unexpected disconnect discards tracking without assuming remote cleanup.

Tool results are bounded before entering the transcript, with binary results described by size. A timeout does not prove the sandbox rejected an operation. Recovery must inspect uncertain effects rather than replaying state changes.

Certain stalled-browser reads may restart the browser and retry once, reporting the reset to the agent. This can discard tabs and page state. Other uncertain browser operations require inspection. Viewer reconnection does not invoke that recovery.

See the user expectations for [viewing and controlling the computer](../../user-expectations/kadan/computer.md).

## Desktop session

The sandbox retains the upstream Xvnc, Openbox and browser supervisor. The image
adds tint2 as a supervised panel using the same VNC enablement and user. Its strut
reserves the bottom 48 pixels, so maximized windows leave the taskbar reachable.
The image generates Openbox configuration from the installed default, preserving
standard window mouse bindings and Alt+Tab while restoring decorations, one
desktop and a small Applications menu. Configuration lives outside `/home/gem`,
so an existing home volume does not hide it or need its settings overwritten.

The pinned browser still owns its existing `.config/browser` profile and CDP
endpoint. Chromium's `--keep-alive-for-test` keeps that process alive after the
last window closes, preventing the upstream supervisor from immediately reopening
a window. The launcher reuses that profile to open a window. This switch is
documented in [Chromium's switch definitions](https://chromium.googlesource.com/chromium/src/+/lkgr/chrome/common/chrome_switches.h).
Existing windows are activated through wmctrl, which also restores minimized
windows. Per-app advisory locks serialize launch requests through window discovery,
with a bounded wait and visible failure dialog. Browser and terminal launchers do
not read credentials, reset profiles or remove browser lock files.

Container replacement can leave Chromium's `SingletonLock`, `SingletonSocket`
and `SingletonCookie` symlinks referring to the old container. Never clear them
automatically at startup. Before removing only those symlinks, an operator must
verify their recorded owner is stopped, no other running container owns the home
volume, and the old socket is absent. Preserve every other profile file. This is
separate from reconnecting a viewer, which needs no profile changes.

## Computer resource limits

`kadan-sandbox` keeps `cpus: 2`, `mem_limit: 4g`, `shm_size: 2gb` and no GPU
device requests. The shared-memory mount is a capacity limit within the container's
memory budget, not an additional allocation. Only this service owns these limits;
changing them does not configure Hikari inference.

Advanced operators can add a local Compose override, for example:

```yaml
# compose.resources.yaml
services:
  kadan-sandbox:
    cpus: 4
    mem_limit: 8g
```

Inspect the resolved configuration before applying it:

```sh
docker compose -f compose.yaml -f compose.resources.yaml config
# After stopping agent/coding work and saving open app state:
docker compose -f compose.yaml -f compose.resources.yaml up -d --no-deps kadan-sandbox
```

Use the same project name and deployment manifests that created the running
computer; custom deployments must adapt their actual manifests instead of
starting a second stack. Recreating the computer interrupts shells, browser tabs,
viewer connections and running jobs. Keep the existing `sandbox_home` volume
mounted at `/home/gem`; never use `down -v` for a resource change. The upstream
startup initializes browser preferences, so persistent profile data is not a
promise to retain every UI preference or open tab. No GPU, additional application
installation or host permission change is implied by increasing CPU or memory.
