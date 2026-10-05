import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH || undefined, headless: true, args: ['--no-sandbox'] })
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } })
    let downloads = 0
    const model = { id: 'wan22-t2v-a14b', repo_id: 'Wan-AI/Wan2.2-T2V-A14B', revision: 'c8c270b13ee05bfa474194ac9fb07a5868a97cea', license: 'apache-2.0', estimated_bytes: 126e9, kind: 'video', display_name: 'Wan2.2 T2V-A14B', inference_available: true, status: 'not_downloaded', downloaded_bytes: 0, total_bytes: null, error: null, context_limit: null, architecture_context_limit: null, license_url: null, license_notice: null }
    await page.route('**/v1/models**', async route => {
      if (route.request().method() === 'POST') {
        assert(new URL(route.request().url()).pathname.endsWith('/wan22-t2v-a14b/download'))
        downloads += 1
        model.status = 'downloading'
      }
      await route.fulfill({ json: { models: [model], selected_model_id: null } })
    })
    await page.route('**/model-lifecycle', route => route.fulfill({ json: { state: 'unloaded', model_id: null } }))
    await page.goto(`${process.env.BASE_URL || 'http://127.0.0.1:5173'}/settings`)
    await page.locator('#model-Video').click()
    await page.getByRole('button', { name: 'Download Wan2.2 T2V-A14B', exact: true }).click()
    await page.getByRole('button', { name: 'Cancel download', exact: true }).waitFor()
    assert.equal(downloads, 1)
    assert.equal(await page.getByRole('dialog').count(), 0)
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
    await page.close()
  }
} finally { await browser.close() }
console.log('Wan T2V Settings download flow passed desktop/mobile')
