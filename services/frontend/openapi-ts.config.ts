import { defineConfig } from '@hey-api/openapi-ts'

// Hikari's generated client; services/hikari/bin/hikari.py generate api refreshes it.
export default defineConfig({
  input: './openapi.json',
  output: './src/hikari/api/generated',
  plugins: ['@hey-api/typescript', '@hey-api/client-fetch', '@hey-api/sdk'],
})
