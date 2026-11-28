import { client } from "../hikari/api/generated/client.gen";
import ServerApp from "./App";
import "../kadan/styles.css";
import "./server.css";

// Hikari's API is served under /hikari; generated paths start at /v1.
client.setConfig({ baseUrl: "/hikari" });

export default ServerApp;
