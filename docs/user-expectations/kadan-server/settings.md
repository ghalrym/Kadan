# Settings

Kadan Server's **Settings** gathers the addresses people need and points to the server's own controls.

- **Connect Kadan** shows this computer's network address with **Copy**, and explains where to paste it in Kadan on another computer (**Settings → Kadan Server → Change → Another computer**). Other devices can use it when network access is allowed.
- **Connect other apps** shows the address that OpenAI-style apps can use to reach the AI models on this computer, with **Copy**. No key is needed. **Technical reference for developers** opens the full API reference.
- **This computer** appears in the Kadan app on the computer that runs Kadan. It shows whether Kadan's services are running, starting, stopped or need attention, with each part (Main service, Agent and AI models) and its status. **Start Kadan's services**, **Stop Kadan's services** and **Restart everything** control them, and each part can be restarted on its own.
- **Server settings** (same place) changes whether other devices on the network can use Kadan, whether its services start with the app, whether Kadan opens when you sign in, where it keeps its data and AI models (with **Choose…**), and the port. Changing network access, the folder or the port restarts the services when saving; moving the folder does not move what is already saved.
- **Logs** (collapsed) shows recent output from Kadan's services, filtered by part, refreshing every few seconds.
- In the Kadan app connected to another computer, **This computer** explains that nothing heavy runs here and how to switch. In a browser, Settings explains that these controls live in the Kadan app on the hosting computer.

Kadan Server provides no built-in protection for these addresses; see [Security and operator responsibility](../security.md).
