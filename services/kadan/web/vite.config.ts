import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API_PROXY_TARGET lets the Compose development override reach kadan-api.
const apiTarget = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8630";
// ws relays the agent computer's VNC stream.
const apiProxy = { target: apiTarget, changeOrigin: false, ws: true };

export default defineConfig({
  plugins: [react()],
  // noVNC (the computer viewer) uses top-level await.
  build: { target: "es2022" },
  optimizeDeps: { esbuildOptions: { target: "es2022" } },
  server: {
    host: "0.0.0.0",
    allowedHosts: true,
    proxy: { "/api": apiProxy, "/openapi.json": apiProxy, "/docs": apiProxy },
    // Poll bind-mounted files when running inside Docker.
    watch: { usePolling: process.env.WATCH_POLLING === "true", interval: 250 },
  },
});
