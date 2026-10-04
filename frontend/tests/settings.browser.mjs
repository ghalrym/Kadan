/**
 * Rendered regression with controlled catalog responses; never downloads weights.
 * Start Vite separately. Run with SETTINGS_TEST_URL=http://127.0.0.1:5173,
 * PLAYWRIGHT_MODULE pointing to an installed Playwright module when not local,
 * and optional PLAYWRIGHT_CHROMIUM_PATH for a system Chromium executable.
 */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH || undefined,
  headless: true,
  args: ['--no-sandbox'],
})
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  const errors = []
  const mutations = []
  page.on('pageerror', error => errors.push(error.message))
  const model = (id, name, status) => ({ id, repo_id: name, revision: 'a'.repeat(40), license: 'apache-2.0', estimated_bytes: 23.5e9, status, downloaded_bytes: 2e9, total_bytes: 23.5e9, error: null, context_limit: null, architecture_context_limit: 262144 })
  const catalog = { models: [model('small', 'nvidia/Qwen3.6-35B-A3B-NVFP4', 'not_downloaded'), model('medium', 'openai/gpt-oss-120b', 'complete'), model('large', 'RedHatAI/GLM-5.3-Flash-NVFP4', 'not_downloaded')], selected_model_id: null }
  let failRead = false
  let failSelection = false
  await page.route('**/v1/models**', async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (request.method() === 'GET' && failRead) return route.fulfill({ status: 503, json: { detail: 'Controlled catalog failure' } })
    if (request.method() !== 'GET') mutations.push([request.method(), path, request.postDataJSON()])
    if (path.endsWith('/selection') && request.method() === 'PUT') {
      if (failSelection) return route.fulfill({ status: 409, json: { detail: 'Unload the running model before changing selection' } })
      catalog.selected_model_id = request.postDataJSON().model_id
    }
    if (path.endsWith('/context')) catalog.models.find(item => path.includes(`/${item.id}/`)).context_limit = request.postDataJSON().context_limit
    if (request.method() === 'POST') { const item = catalog.models.find(item => path.includes(`/${item.id}/`)); item.status = 'downloading'; item.error = null }
    if (request.method() === 'DELETE') catalog.models.find(item => path.includes(`/${item.id}/`)).status = 'cancelled'
    return route.fulfill({ json: catalog })
  })
  await page.goto(`${process.env.SETTINGS_TEST_URL || 'http://127.0.0.1:5173'}/settings`)
  const trigger = page.getByRole('button', { name: 'Language model', exact: true })
  await trigger.waitFor()
  for (const name of ['Image generation', 'Video generation', 'Text to speech', 'Speech to text']) assert(await page.getByLabel(name, { exact: true }).isDisabled())
  assert(await page.getByRole('switch', { name: 'Whisper S1 Mini formatting' }).isDisabled())
  assert(await page.getByLabel('Max context length (tokens)').isDisabled())
  const open = async () => { await trigger.click(); await page.getByRole('dialog', { name: 'Language model options' }).waitFor() }
  await open()
  const dialog = page.getByRole('dialog', { name: 'Language model options' })
  assert(await dialog.getByRole('button', { name: /Qwen3.6.*not downloaded/ }).isDisabled())
  const download = dialog.getByRole('button', { name: 'Download Qwen3.6-35B-A3B-NVFP4', exact: true })
  assert(await download.evaluate(element => element === document.activeElement))
  await page.keyboard.press('End')
  assert(await dialog.getByRole('button', { name: 'Close model options' }).evaluate(element => element === document.activeElement))
  await page.keyboard.press('Escape')
  assert(await trigger.evaluate(element => element === document.activeElement))
  await open()
  await page.getByRole('heading', { name: 'Models', exact: true }).click()
  assert.equal(await dialog.count(), 0)
  await open()
  await dialog.getByRole('button', { name: 'Close model options' }).focus()
  await page.keyboard.press('Tab')
  await dialog.waitFor({ state: 'hidden' })
  await open()
  await download.click()
  await page.getByRole('progressbar', { name: 'small download progress' }).waitFor()
  assert(await trigger.evaluate(element => element === document.activeElement))
  assert.equal(await page.getByRole('progressbar').count(), 1)
  await open()
  await page.screenshot({ path: '/tmp/kadan-settings-compact-desktop-progress.png', fullPage: true })
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: 'Cancel download', exact: true }).click()
  await page.getByRole('button', { name: 'Retry download', exact: true }).waitFor()
  await page.getByRole('button', { name: 'Retry download', exact: true }).click()
  catalog.models[0].status = 'failed'; catalog.models[0].error = 'Controlled download failure'
  await page.getByRole('alert').filter({ hasText: 'Controlled download failure' }).waitFor()
  await page.getByRole('button', { name: 'Retry download', exact: true }).click()
  catalog.models[0].status = 'complete'
  await page.getByRole('progressbar').waitFor({ state: 'hidden' })
  await open()
  await dialog.getByRole('button', { name: /Qwen3.6.*complete/ }).click()
  await open()
  await dialog.getByRole('button', { name: /Qwen3.6.*Selected/ }).waitFor()
  await page.keyboard.press('Escape')
  assert(await trigger.evaluate(element => element === document.activeElement))
  catalog.selected_model_id = 'medium'
  await page.waitForFunction(() => document.querySelector('#model-LLM')?.textContent.includes('gpt-oss-120b'))
  catalog.selected_model_id = 'small'
  await page.waitForFunction(() => document.querySelector('#model-LLM')?.textContent.includes('Qwen3.6'))
  const context = page.getByLabel('Max context length (tokens)')
  await context.fill('262145')
  assert(await page.getByRole('button', { name: 'Save context window' }).isDisabled())
  await context.fill('4096')
  await page.getByRole('button', { name: 'Save context window' }).click()
  await page.waitForFunction(() => document.querySelector('.model-context input')?.value === '4096' && document.querySelector('.model-context button')?.disabled)
  failSelection = true
  await open()
  await dialog.getByRole('button', { name: /gpt-oss-120b.*complete/ }).click()
  await page.getByRole('alert').filter({ hasText: 'Unload the running model' }).waitFor()
  assert.equal(catalog.selected_model_id, 'small')
  assert.match(await trigger.innerText(), /Qwen3.6/)
  failRead = true
  await page.getByRole('button', { name: 'Refresh', exact: true }).waitFor()
  failRead = false
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await page.getByRole('button', { name: 'Refresh', exact: true }).waitFor({ state: 'hidden' })
  await page.reload()
  await page.waitForFunction(() => document.querySelector('.model-context input')?.value === '4096' && document.querySelector('.model-context button')?.disabled)
  assert.doesNotMatch(await page.locator('.settings-layout').innerText(), /Saved:|Blank uses|larger windows|Selection does not|Not selected|Model card|apache-2.0|disk reserve|providers are implemented|Download a model before/)
  await page.screenshot({ path: '/tmp/kadan-settings-compact-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 768, height: 1024 })
  assert.equal(await page.locator('.model-llm-details').evaluate(element => getComputedStyle(element).gridColumnStart), '1')
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
  await page.screenshot({ path: '/tmp/kadan-settings-compact-tablet.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await open()
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
  await page.screenshot({ path: '/tmp/kadan-settings-compact-mobile.png', fullPage: true })
  await page.keyboard.press('Home')
  await page.keyboard.press('Tab')
  await page.keyboard.press('Escape')
  assert.equal(await dialog.count(), 0)
  assert.deepEqual(errors, [])
  assert(mutations.some(([method, path]) => method === 'DELETE' && path.endsWith('/small/download')))
  assert(mutations.some(([method, path, body]) => method === 'PUT' && path.endsWith('/context') && body.context_limit === 4096))
  console.log('PASS compact settings: original rows, download/cancel/retry/progress/selection/context/errors, keyboard, desktop/mobile, no page errors')
} finally {
  await browser.close()
}
