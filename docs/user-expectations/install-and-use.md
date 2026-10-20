# Install and use

Users expect to start Kadan with one Docker Compose file and zero custom configuration. Once the required hardware, drivers, Docker and NVIDIA Container Toolkit are available, running this command from the repository root must start the baseline system:

```sh
docker compose up --build -d
```

Startup must work without creating a `.env` file, generating secrets, selecting build targets or combining Compose files. Every parameter needed by the baseline system must have a working default. The browser interfaces must be available at their documented default addresses, including from other computers on the trusted network via the host IP. Ports 8630, 8620 and 8088 bind to 0.0.0.0; operator links must use the browser host rather than the remote client's localhost. The direct model API and sandbox remain private. DNS names may require an explicit host allowlist; an unrestricted host allowlist is not the default.

Users can optionally create a `.env` file to override default parameters, such as ports, credentials and remote worker connections. `.env.example` documents the available overrides. Customization must remain optional for the baseline system.

Hot reload is the standard behavior. Source edits must reach the running services through automatic reloads or incremental worker rebuilds, without selecting a separate mode.

Starting the system and downloading model checkpoints are separate steps. Users select and download the checkpoints they need through Hikari Settings after startup; startup must not require manual model configuration or run inference.
