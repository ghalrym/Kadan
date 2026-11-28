# Install and use

Kadan ships as two desktop applications that run directly on your computer without a separate database:

- **Kadan Server** runs everything Kadan needs in the background (agent conversations, storage, and local AI models) and lives in your system tray. Its window has three tabs: **Kadan Server** (the [management pages](kadan-server/overview.md): Home, AI models, Your agent, Activity and Settings), **This computer** (connection addresses, background services, hardware, updates and server settings), and **Logs**. The management pages also open in any browser at the server's address followed by `/server`.
- **Kadan** is the everyday desktop app for working with your agent. Open **Server settings** in the sidebar to enter or switch the host address of any **Kadan Server** running on the same computer or on your local network, test the connection, and save multiple servers.

Starting Kadan Server and downloading model checkpoints are separate steps. Open **AI models** in Kadan Server after startup to download the models you need; startup never runs inference or acquires models automatically. Downloads, license acknowledgements and subscription choices all happen there.

Kadan Server stores its settings, database, files, and downloaded models in a single data folder that you can change from **This computer → Server Settings**. Stopping or restarting Kadan Server preserves all saved data.

Upgrades and service restarts can interrupt active requests. Saved conversations and durable work recover according to their feature's rules, while in-flight inference cancels cleanly when Hikari restarts. Refresh saved state before retrying uncertain work.

You can ask your agent at any time to check for Kadan updates or to update Kadan for you. When a newer release is available, the agent reports what is available and offers to install it. If you confirm, Kadan downloads and verifies the update first before restarting; if the download or verification fails, nothing restarts and your current version keeps running. On supported desktop installations (Linux AppImage, Windows installer, and macOS app bundle), Kadan Server—and Kadan Desktop if connected—install the update and restart automatically while preserving your settings, conversations, memories, files, and downloaded models. Installations that cannot self-update (such as manual package or developer installations) report that plainly so you can update them manually.

Back up your data folder before replacing an installation or moving machines. [Tsuchi backups](tsuchi/backup.md) preserve agent memory and files; downloaded models can always be downloaded again from **AI models**. No automatic cloud backup is provided.
