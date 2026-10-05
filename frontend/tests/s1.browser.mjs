/** Render native Whisper selection/recording with controlled responses; no model runs. */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH, headless: true,
  args: ['--no-sandbox', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] })
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, permissions: ['microphone'] })
    const page = await context.newPage()
    const errors = []; page.on('pageerror', error => errors.push(error.message))
    let expectedFormatting = false; let selected = 'large-v3'; let posts = 0; let fail = true
    const names = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo']
    const model = { id: 's1-mini', kind: 'formatting', display_name: 'S1-mini by Superwhisper', repo_id: 'superwhisper/s1-mini', revision: 'pinned', license: 'Apache-2.0', estimated_bytes: 1520000000, status: 'not_downloaded', downloaded_bytes: 0, total_bytes: 0, error: null }
    const catalog = () => ({ models: [model], selected_model_id: null })
    await page.route('**/v1/models', route => route.fulfill({ json: catalog() }))
    await page.route('**/v1/models/s1-mini/download', async route => {
      model.status = route.request().method() === 'DELETE' ? 'cancelled' : 'downloading'
      await route.fulfill({ json: catalog() })
    })
    await page.route('**/model-lifecycle', route => route.fulfill({ json: { state: 'unloaded', model_id: null } }))
    await page.route('**/v1/audio/transcriptions/models', async route => {
      if (route.request().method() === 'PUT') selected = route.request().postDataJSON().model
      await route.fulfill({ json: { models: names, selected } })
    })
    await page.route('**/v1/audio/transcriptions', async route => {
      posts++
      const body = route.request().postDataJSON()
      assert.equal(body.formatting, expectedFormatting)
      assert.ok(body.audio.startsWith('data:audio/wav;base64,'))
      const wav = Buffer.from(body.audio.split(',')[1], 'base64')
      assert.equal(wav.readUInt32LE(24), 16000)
      assert.equal(wav.readUInt16LE(22), 1)
      await route.fulfill(fail ? { status: 503, json: { detail: 'Download this Whisper checkpoint.' } } : { json: { text: 'A controlled transcript.', formatting_status: 'unsupported_language' } })
    })
    const url = process.env.WHISPER_TEST_URL || 'http://127.0.0.1:15237'
    await page.goto(`${url}/settings`)
    const toggle = page.getByRole('switch', { name: 'S1-mini by Superwhisper formatting' })
    assert.equal(await toggle.getAttribute('aria-checked'), 'true')
    await toggle.click()
    await page.reload()
    assert.equal(await toggle.getAttribute('aria-checked'), 'false')
    assert.equal(await toggle.locator('span').evaluate(element => getComputedStyle(element).left), '2px')
    await page.locator('#model-Formatting').click()
    await page.getByRole('button', { name: 'Download S1-mini by Superwhisper', exact: true }).click()
    await page.getByRole('progressbar', { name: 'S1-mini download progress' }).waitFor()
    await page.getByRole('button', { name: 'Cancel download', exact: true }).click()
    await page.getByRole('button', { name: 'Retry download', exact: true }).click()
    await page.getByRole('progressbar', { name: 'S1-mini download progress' }).waitFor()
    await page.screenshot({ path: `/tmp/s1-settings-${width}.png`, fullPage: true })
    await page.goto(`${url}/stt`)
    await page.getByRole('button', { name: 'Start recording' }).click()
    await page.getByRole('button', { name: 'Stop recording' }).waitFor()
    await page.waitForTimeout(500)
    await page.getByRole('button', { name: 'Stop recording' }).click()
    await page.getByRole('button', { name: 'Submit', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'Download this Whisper checkpoint.' }).waitFor()
    assert.equal(posts, 1)
    fail = false
    expectedFormatting = true
    await page.evaluate(() => localStorage.setItem('kadan.s1Formatting', 'true'))
    await page.getByRole('button', { name: 'Submit', exact: true }).click()
    await page.getByRole('status').filter({ hasText: 'A controlled transcript.' }).waitFor()
    assert.equal(posts, 2)
    await page.getByRole('status').filter({ hasText: 'S1-mini supports English only' }).waitFor()
    await page.screenshot({ path: `/tmp/s1-recording-${width}.png`, fullPage: true })
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
    await page.getByRole('button', { name: 'Discard' }).click()
    assert.equal(await page.getByRole('status').count(), 0)
    assert.deepEqual(errors, [])
    await context.close()
  }
  console.log('S1 desktop/mobile preference, reload persistence, request flags and fallback passed')
} finally { await browser.close() }
