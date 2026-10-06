/** Rendered native video job states using HTTP fixtures, never model inference. */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH || undefined, headless: true, args: ['--no-sandbox'] })
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } })
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    let jobs = []
    let cancelled = 0
    await page.route('**/v1/videos**', async route => {
      const method = route.request().method()
      const path = new URL(route.request().url()).pathname
      if (method === 'POST') {
        const body = route.request().postDataJSON()
        jobs = [{ id: 'fixture', prompt: body.prompt, duration: '8s', resolution: '720p', aspect: 'wide', fps: '24', time: 'now', progress: 0, status: 'Rendering', thumbnail: '', progressText: 'Rendering', output_url: null, error: null }]
        return route.fulfill({ json: { job: jobs[0] } })
      }
      if (method === 'DELETE') {
        cancelled++
        jobs[0] = { ...jobs[0], status: 'Cancelled', progressText: 'Cancelled' }
      }
      return route.fulfill({ json: path.endsWith('/videos') ? { jobs } : { job: jobs[0] } })
    })
    await page.goto(`${process.env.VIDEO_TEST_URL || 'http://127.0.0.1:15239'}/video`)
    await page.getByText('No video jobs.', { exact: true }).waitFor()
    await page.getByLabel('Prompt', { exact: true }).fill('A forest canopy')
    await page.getByRole('button', { name: 'Queue video', exact: true }).click()
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()
    await page.locator('.badge').filter({ hasText: 'Cancelled' }).waitFor()
    assert.equal(cancelled, 1)
    jobs[0] = { ...jobs[0], status: 'Failed', error: 'Native worker failed', progressText: 'Failed' }
    await page.getByRole('button', { name: 'Refresh queue' }).click()
    await page.getByRole('alert').filter({ hasText: 'Native worker failed' }).waitFor()
    jobs[0] = { ...jobs[0], status: 'Done', error: null, progress: 100, progressText: 'Complete', output_url: '/v1/videos/fixture/content' }
    await page.getByRole('button', { name: 'Refresh queue' }).click()
    await page.getByLabel('Generated video').waitFor()
    assert.equal(await page.getByLabel('Generated video').getAttribute('src'), '/v1/videos/fixture/content')
    assert.deepEqual(errors, [])
    await page.screenshot({ path: `/tmp/kadan-native-video-${width}.png`, fullPage: true })
    await page.close()
  }
  console.log('Native video rendered desktop/mobile cancel, failure and playback states passed')
} finally { await browser.close() }
