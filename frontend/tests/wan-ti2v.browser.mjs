/** TI2V conditioning upload and generation contract; no model runs. */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/opt/codex/runtimes/cua/lib/node_modules/playwright')
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] })
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } })
    let failUpload = true, uploads = 0
    const requests = []
    await page.route('**/v1/videos**', async route => {
      const request = route.request(), path = new URL(request.url()).pathname
      if (path.endsWith('/inputs')) {
        uploads++
        assert.equal(request.headers()['content-type'], 'image/png')
        assert.equal(new URL(request.url()).searchParams.get('kind'), 'image')
        return route.fulfill({ status: failUpload ? 422 : 200, json: { id: 'a'.repeat(32) } })
      }
      if (request.method() === 'POST') {
        requests.push(request.postDataJSON())
        return route.fulfill({ status: 503, json: { detail: 'Worker unavailable' } })
      }
      return route.fulfill({ json: { jobs: [] } })
    })
    await page.goto(process.env.WAN_UI_URL || 'http://127.0.0.1:15248/video')
    await page.getByLabel('Model', { exact: true }).selectOption('wan22-ti2v-5b')
    await page.getByLabel('Prompt', { exact: true }).fill('A forest canopy')
    await page.getByLabel('Reference image (optional)', { exact: true }).setInputFiles({ name: 'reference.png', mimeType: 'image/png', buffer: Buffer.from('tiny fixture') })
    await page.getByRole('button', { name: 'Queue video', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'Reference image upload failed' }).waitFor()
    assert.equal(requests.length, 0)
    failUpload = false
    await page.getByRole('button', { name: 'Queue video', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'Video generation is unavailable' }).waitFor()
    assert.equal(uploads, 2)
    assert.equal(requests.length, 1)
    assert.equal(requests[0].image_id, 'a'.repeat(32))
    assert.equal(requests[0].model, 'wan22-ti2v-5b')
    assert.equal(requests[0].fps, 24)
    assert.equal(requests[0].resolution, '720p')
    assert.equal(await page.getByRole('button', { name: '1080p', exact: true }).count(), 0)
    await page.screenshot({ path: `/tmp/kadan-wan-ti2v-${width}.png`, fullPage: true })
    await page.close()
    console.log(`TI2V image upload failure/retry/payload passed at ${width}px`)
  }
} finally { await browser.close() }
