# Security and operator responsibility

Kadan is intentionally bare bones. The only built-in security is basic API-key authentication where needed. Every such key defaults to the public value `insecure-api-key`, so the default configuration provides no meaningful protection. Users are responsible for adding and maintaining any protection they require.

Users are responsible for securing the machines and environments where they run Kadan. This responsibility also applies to deployments at scale; the project does not provide security for those deployments on the user's behalf.

While the project is still being fleshed out, development speed takes priority. Security work adds development overhead and is the lowest priority during this phase. Users who need security are expected to configure it themselves as part of operating the system.

Hikari workers and the agent computer use configurable API keys, both defaulting to `insecure-api-key`. Startup does not require generating credentials. See the [installation configuration](../../README.md) for operator settings.

The baseline has no host or origin allowlists, separate sandbox network, or container hardening requirements. Published ports listen on all host interfaces. The application interfaces have no login or access-control system. Operators decide how to restrict access to their deployment.
