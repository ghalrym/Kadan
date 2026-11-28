import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { scopeSectionCss } from './config/scopeSectionCss.ts'

/** One development server for Kadan and Kadan Server pages.
 * API_URL is the combined API, which serves each service under its own prefix.
 * This does not configure production routing.
 */
const api = process.env.API_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  css: {
    postcss: {
      plugins: [scopeSectionCss(path.resolve(import.meta.dirname, 'src'), ['kadan'])],
    },
  },
  server: {
    host: '0.0.0.0',
    allowedHosts: true,
    proxy: {
      // OAuth 2.1 discovery for ChatGPT / external Group Chat MCP clients.
      '/.well-known': api,
      // ws relays Kadan voice calls.
      '/kadan/api': { target: api, ws: true },
      '/kadan/mcp': api,
      '/tsuchi/api': api,
      '/hikari/v1': { target: api, ws: true },
      '/hikari/model-lifecycle': api,
      '/hikari/health': api,
      '/hikari/openapi.json': api,
      '/hikari/docs': api,
    },
    watch: {
      // Optional polling for filesystems where native watchers miss edits.
      usePolling: process.env.WATCH_POLLING === 'true',
      interval: 250,
    },
  },
})
