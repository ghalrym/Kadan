import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
      '/health': process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
    },
    watch: {
      // Poll bind-mounted files when running inside Docker.
      usePolling: process.env.WATCH_POLLING === 'true',
      interval: 250,
    },
  },
})
