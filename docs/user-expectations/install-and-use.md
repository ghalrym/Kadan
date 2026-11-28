# Install and use

Kadan is one desktop app that runs directly on your computer without a separate database. In **Settings → Kadan Server → Change**, it offers two choices:

- **This computer:** Kadan runs everything here, including its AI models. Closing the window keeps it running in the system tray so your agent and other devices can still use it; choose **Quit Kadan** from the tray to stop it. This is the default for a new install.
- **Another computer:** Kadan uses the services of Kadan running on another computer, such as a home server with a strong graphics card. Paste that computer's address (shown on its Kadan Server **Home** and **Settings**), test the connection and save it; saved addresses can be picked again later. Nothing heavy runs on this computer, and closing the window quits Kadan.

Switching takes effect immediately; nothing needs restarting. Kadan Server's [management pages](kadan-server/overview.md) (Home, AI models, Your agent, Activity and Settings) open from **Settings → Kadan Server → Open Kadan Server**, and in any browser at the hosting computer's address followed by `/server`. When this computer hosts, Kadan Server's Settings also starts and stops Kadan's services, restarts parts of it, changes network access, the port, the data folder and starting with the computer, and shows logs.

Kadan installed before this change keeps doing what it did: a computer that ran Kadan Server keeps hosting with its existing data, and a Kadan app pointed at another computer keeps connecting to it.

Starting Kadan Server and downloading model checkpoints are separate steps. Open **AI models** in Kadan Server after startup to download the models you need; startup never runs inference or acquires models automatically. Downloads, license acknowledgements and subscription choices all happen there.

Kadan Server stores its settings, database, files, and downloaded models in a single data folder that you can change from Kadan Server's **Settings** on the hosting computer. Stopping or restarting Kadan Server preserves all saved data.

Upgrades and service restarts can interrupt active requests. Saved conversations and durable work recover according to their feature's rules, while in-flight inference cancels cleanly when Hikari restarts. Refresh saved state before retrying uncertain work.

You can ask your agent at any time to check for Kadan updates or to update Kadan for you. When a newer release is available, the agent reports what is available and offers to install it. If you confirm, Kadan downloads and verifies the update first before restarting; if the download or verification fails, nothing restarts and your current version keeps running. On supported desktop installations (Linux AppImage, Windows installer, and macOS app bundle), Kadan on the hosting computer, and any Kadan apps connected to it, install the update and restart automatically while preserving your settings, conversations, memories, files, and downloaded models. Installations that cannot self-update (such as manual package or developer installations) report that plainly so you can update them manually.

Back up your data folder before replacing an installation or moving machines. [Tsuchi backups](tsuchi/backup.md) preserve agent memory and files; downloaded models can always be downloaded again from **AI models**. No automatic cloud backup is provided.
