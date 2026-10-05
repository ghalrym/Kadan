import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH || undefined, headless: true, args: ['--no-sandbox'] })
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } })
    const requests = []
    await page.route('**/v1/videos**', async route => {
      const url = new URL(route.request().url())
      if (url.pathname.endsWith('/inputs')) {
        assert.equal(route.request().method(), 'POST')
        const kind = url.searchParams.get('kind')
        assert(['image', 'video'].includes(kind))
        return route.fulfill({ json: { id: (kind === 'image' ? 'a' : 'b').repeat(32) } })
      }
      if (url.pathname.endsWith('/generations')) {
        requests.push(route.request().postDataJSON())
        return route.fulfill({ status: 503, json: { detail: 'Worker unavailable' } })
      }
      return route.fulfill({ json: { jobs: [] } })
    })
    await page.goto(`${process.env.BASE_URL || 'http://127.0.0.1:5173'}/video`)
    await page.getByLabel('Prompt', { exact: true }).fill('A dancer')
    assert(await page.getByRole('button', { name: 'Queue video' }).isDisabled())
    await page.getByLabel('Reference image').setInputFiles({ name: 'reference.png', mimeType: 'image/png', buffer: Buffer.from('fixture') })
    await page.getByLabel('Driving video').setInputFiles({ name: 'driving.mp4', mimeType: 'video/mp4', buffer: Buffer.from('fixture') })
    for (const mode of ['animate', 'replace']) {
      await page.getByLabel('Animation mode', { exact: true }).selectOption(mode)
      await page.getByRole('button', { name: 'Queue video' }).click()
      await page.getByText('Video generation is unavailable.', { exact: false }).waitFor()
      const body = requests.at(-1)
      assert.equal(body.model, 'wan22-animate-14b')
      assert.equal(body.animation_mode, mode)
      assert.equal(body.image_id, 'a'.repeat(32)); assert.equal(body.video_id, 'b'.repeat(32))
      assert.equal(body.fps, 30); assert.equal(body.resolution, '720p')
    }
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
    if (process.env.SCREENSHOT_DIR) await page.screenshot({ path: `${process.env.SCREENSHOT_DIR}/wan-animate-${width}.png`, fullPage: true })
    await page.close()
  }
} finally { await browser.close() }
console.log('Wan Animate upload, mode and failure UI passed desktop/mobile')
