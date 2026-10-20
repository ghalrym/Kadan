import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

/** Development proxies keep API and documentation links on the frontend origin.
 * API_PROXY_TARGET chooses the backend; this does not configure production routing.
 */
const allowedHosts: unknown = JSON.parse(process.env.ALLOWED_HOSTS ?? "[]");
if (!Array.isArray(allowedHosts) || !allowedHosts.every((host): host is string => typeof host === "string")) {
  throw new Error("ALLOWED_HOSTS must be a JSON array of hostnames");
}

export default defineConfig({
  plugins: [react()],
  server: {
    // Vite already accepts IP literals; DNS names require explicit configuration.
    allowedHosts,
    proxy: {
      '/v1': process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
      '/model-lifecycle': process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
      '/health': process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
      '/openapi.json': process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
      '/docs': process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
    },
    watch: {
      // Poll bind-mounted files when running inside Docker.
      usePolling: process.env.WATCH_POLLING === 'true',
      interval: 250,
    },
  },
})
