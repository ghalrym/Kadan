/** Exercise image-conditioned submission with controlled API responses, not model weights. */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH, headless: true, args: ['--no-sandbox'] })
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1100 } })
    const errors = []; page.on('pageerror', error => errors.push(error.message))
    let uploads = 0; let generations = 0; let failUpload = true; let failGenerate = true
    const imageId = 'a'.repeat(32)
    await page.route('**/v1/videos', route => route.fulfill({ json: { jobs: [] } }))
    await page.route('**/v1/videos/inputs?kind=image', route => {
      uploads++
      assert.equal(route.request().headers()['content-type'], 'image/png')
      assert.ok(route.request().postDataBuffer().length > 30)
      return route.fulfill(failUpload ? { status: 422, json: { detail: 'Controlled upload failure' } } : { json: { id: imageId } })
    })
    await page.route('**/v1/videos/generations', route => {
      generations++
      const body = route.request().postDataJSON()
      assert.equal(body.model, 'wan22-i2v-a14b')
      assert.equal(body.image_id, imageId)
      assert.equal(body.fps, 16)
      assert.equal(body.resolution, '720p')
      assert.ok(!('image_path' in body))
      return route.fulfill(failGenerate ? { status: 503, json: { detail: 'Unavailable' } } : { status: 202, json: { job: {
        id: 'controlled-i2v', prompt: body.prompt, duration: '8s', resolution: '720p', aspect: 'wide', fps: '16',
        progress: 0, time: 'now', status: 'Queued', thumbnail: '', progressText: 'Queued',
      } } })
    })
    await page.goto(process.env.WAN_I2V_TEST_URL || 'http://127.0.0.1:15247/video')
    await page.getByLabel('Model', { exact: true }).selectOption('wan22-i2v-a14b')
    await page.getByLabel('Prompt', { exact: true }).fill('A bird begins to fly')
    assert.equal(await page.getByRole('button', { name: 'Queue video', exact: true }).isDisabled(), true)
    await page.getByLabel('Input image', { exact: true }).setInputFiles({ name: 'input.png', mimeType: 'image/png', buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==', 'base64') })
    await page.getByRole('button', { name: 'Queue video', exact: true }).click()
    await page.getByText('Controlled upload failure', { exact: true }).waitFor()
    assert.equal(generations, 0)
    failUpload = false
    await page.getByRole('button', { name: 'Queue video', exact: true }).click()
    await page.getByText('Video generation is unavailable. Check the downloaded model and worker configuration.', { exact: true }).waitFor()
    assert.equal(generations, 1)
    failGenerate = false
    await page.getByRole('button', { name: 'Queue video', exact: true }).click()
    await page.getByText('1 pending · 0 done', { exact: true }).waitFor()
    assert.equal(uploads, 3)
    assert.equal(generations, 2)
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
    assert.deepEqual(errors, [])
    await page.screenshot({ path: `/tmp/wan-i2v-${width}.png`, fullPage: true })
    await page.close()
  }
  console.log('Wan I2V desktop/mobile image upload, required image, error/retry and conditioned submission passed')
} finally { await browser.close() }
