import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API_PROXY_TARGET lets the Compose development override reach tsuchi-api.
const apiTarget = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8620";
// Forward the browser's Host header like nginx does; the API rejects unknown hosts.
const apiProxy = { target: apiTarget, changeOrigin: false };

const allowedHosts: unknown = JSON.parse(process.env.ALLOWED_HOSTS ?? "[]");
if (!Array.isArray(allowedHosts) || !allowedHosts.every((host): host is string => typeof host === "string")) {
  throw new Error("ALLOWED_HOSTS must be a JSON array of hostnames");
}

export default defineConfig({
  plugins: [react()],
  server: {
    // Vite already accepts IP literals; DNS names require explicit configuration.
    allowedHosts,
    host: "127.0.0.1",
    proxy: { "/api": apiProxy, "/openapi.json": apiProxy, "/docs": apiProxy },
    // Poll bind-mounted files when running inside Docker.
    watch: { usePolling: process.env.WATCH_POLLING === "true", interval: 250 },
  },
});
