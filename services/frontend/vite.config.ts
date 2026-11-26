import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { scopeSectionCss } from './config/scopeSectionCss.ts'

/** One development server for Kadan, Tsuchi and Hikari pages.
 * Each *_API_URL chooses a backend; this does not configure production routing.
 */
const kadan = process.env.KADAN_API_URL ?? 'http://127.0.0.1:8630'
const tsuchi = process.env.TSUCHI_API_URL ?? 'http://127.0.0.1:8620'
const hikari = process.env.HIKARI_API_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  css: {
    postcss: {
      plugins: [scopeSectionCss(path.resolve(import.meta.dirname, 'src'), ['kadan', 'tsuchi', 'hikari'])],
    },
  },
  server: {
    host: '0.0.0.0',
    allowedHosts: true,
    proxy: {
      // Tsuchi and Kadan both serve /api, so Tsuchi's is reached under /tsuchi.
      '/tsuchi/api': {
        target: tsuchi,
        changeOrigin: false,
        rewrite: (url) => url.replace(/^\/tsuchi/, ''),
      },
      // ws relays voice call streams.
      '/api': { target: kadan, changeOrigin: false, ws: true },
      '/v1': { target: hikari, ws: true },
      '/model-lifecycle': hikari,
      '/health': hikari,
      '/openapi.json': hikari,
      '/docs': hikari,
    },
    watch: {
      // Poll bind-mounted files when running inside Docker.
      usePolling: process.env.WATCH_POLLING === 'true',
      interval: 250,
    },
  },
})
