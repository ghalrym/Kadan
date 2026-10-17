import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API_PROXY_TARGET lets the Compose development override reach tsuchi-api.
const apiTarget = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8620";
// Forward the browser's Host header like nginx does; the API rejects unknown hosts.
const apiProxy = { target: apiTarget, changeOrigin: false };

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    proxy: { "/api": apiProxy, "/openapi.json": apiProxy, "/docs": apiProxy },
    // Poll bind-mounted files when running inside Docker.
    watch: { usePolling: process.env.WATCH_POLLING === "true", interval: 250 },
  },
});
