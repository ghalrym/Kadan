import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      // Poll bind-mounted files when running inside Docker.
      usePolling: process.env.WATCH_POLLING === 'true',
      interval: 250,
    },
  },
})
