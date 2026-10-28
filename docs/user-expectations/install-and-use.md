# Install and use

Users expect to start Kadan with one Docker Compose file and zero custom configuration. Once the required hardware, drivers, Docker and NVIDIA Container Toolkit are available, running this command from the repository root must start the baseline system:

```sh
docker compose up --build -d
```

Startup must work without creating a `.env` file, generating secrets, selecting build targets or combining Compose files. Every parameter needed by the baseline system must have a working default. The browser interfaces must be available at their documented default addresses.

Users can optionally create a `.env` file to override default parameters, such as ports, credentials and remote worker connections. `.env.example` documents the available overrides. Customization must remain optional for the baseline system.

Hot reload is the standard behavior. Source edits must reach the running services through automatic reloads or incremental worker rebuilds, without selecting a separate mode.

Starting the system and downloading model checkpoints are separate steps. Users select and download the checkpoints they need through Hikari Settings after startup; startup must not require manual model configuration or run inference.

Kadan is the everyday agent interface; Tsuchi manages saved content and Hikari manages inference. Kadan reports [service availability](kadan/service-status.md). A running interface does not establish that models, inference workers or agent runners are ready. Optional [remote workers](hikari/remote-workers.md) and [performance tracing](tracing.md) require additional operator configuration.

Stopping, rebuilding or replacing containers preserves data when the same persistent stores and deployment identity are retained. Deleting data volumes removes that protection. Existing installations must keep their storage settings when upgrading; a fresh project or volume can appear empty while the old data still exists elsewhere. The [startup and upgrade procedure](../../README.md#existing-installations-and-data) explains the operator steps.

Upgrades and service restarts can interrupt requests. Saved conversations and durable work can recover according to their feature's rules, while browser-only playground state and temporary monitoring records can be lost. Hikari API restarts cancel in-flight inference; recovery does not promise transparent continuation of every request. Refresh saved state before retrying uncertain work.

Back up data before replacing an installation, and inspect a restored copy before relying on it. [Tsuchi backups](tsuchi/backup.md) preserve its store; model downloads, deployment settings and the agent computer's files have separate lifetimes. No automatic backup or complete-system recovery is provided.
