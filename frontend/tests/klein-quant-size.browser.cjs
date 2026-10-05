const { chromium } = require('/opt/codex/runtimes/cua/lib/node_modules/playwright')
const assert = require('node:assert/strict')
;(async () => {
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] })
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } })
      let total = 0
      await page.route('**/model-lifecycle', route => route.fulfill({ json: { state: 'unloaded', model_id: null, error: null } }))
      await page.route('**/v1/models', route => route.fulfill({ json: {
        selected_model_id: null, selected_image_model_id: null,
        models: [{ id: 'synthetic-quant', kind: 'image', repo_id: 'fixture/quant', display_name: 'Synthetic quant fixture',
          revision: 'fixture', license: 'apache-2.0', estimated_bytes: 0, total_bytes: total, downloaded_bytes: 0,
          inference_available: true, status: 'available', error: null, context_limit: null, architecture_context_limit: null,
          license_notice: null, license_url: null }],
      } }))
      await page.goto('http://127.0.0.1:15249/settings')
      await page.locator('#model-Image').click()
      await page.getByText('Checking size', { exact: true }).waitFor()
      assert.equal(await page.getByText('0.0 GB', { exact: true }).count(), 0)
      total = 12e9
      await page.reload()
      await page.locator('#model-Image').click()
      await page.getByText('12.0 GB', { exact: true }).waitFor()
      assert.equal(await page.getByText('Checking size', { exact: true }).count(), 0)
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
      await page.screenshot({ path: `/tmp/klein-quant-size-${width}.png`, fullPage: true })
      await page.close()
    }
    console.log('Unknown size and manifest total rendered correctly at desktop and mobile widths')
  } finally { await browser.close() }
})()
