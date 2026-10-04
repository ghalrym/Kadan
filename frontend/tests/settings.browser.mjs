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
  const model = (id, name, status) => ({ id, repo_id: name, revision: 'a'.repeat(40), license: 'apache-2.0', estimated_bytes: 23.5e9, status, downloaded_bytes: 2e9, total_bytes: 23.5e9, error: null, context_limit: 65536, architecture_context_limit: 262144 })
  const catalog = { models: [model('small', 'nvidia/Qwen3.6-35B-A3B-NVFP4', 'not_downloaded'), model('medium', 'openai/gpt-oss-120b', 'complete'), model('large', 'RedHatAI/GLM-5.3-Flash-NVFP4', 'not_downloaded')], selected_model_id: null }
  let lifecycle = { state: 'unloaded', model_id: null, configured_context_limit: null, error: null }
  let failLoad = false
  let failLifecycleRead = false
  let releaseLoad
  await page.route('**/v1/models**', async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (request.method() !== 'GET') mutations.push([request.method(), path, request.postDataJSON()])
    assert(!path.endsWith('/selection') && !path.endsWith('/context'), 'Only unified Load may persist selection/context')
    if (request.method() === 'POST') { const item = catalog.models.find(item => path.includes(`/${item.id}/`)); item.status = 'downloading'; item.error = null }
    if (request.method() === 'DELETE') catalog.models.find(item => path.includes(`/${item.id}/`)).status = 'cancelled'
    return route.fulfill({ json: catalog })
  })
  await page.route('**/model-lifecycle**', async route => {
    if (route.request().method() === 'POST') {
      const body = route.request().postDataJSON()
      mutations.push(['POST', '/model-lifecycle/load', body])
      if (failLoad) return route.fulfill({ status: 409, json: { detail: 'Controlled busy load failure' } })
      await new Promise(resolve => { releaseLoad = resolve })
      catalog.selected_model_id = body.model_id
      catalog.models.find(model => model.id === body.model_id).context_limit = body.context_limit
      lifecycle = { state: 'loading', model_id: body.model_id, configured_context_limit: body.context_limit, error: null }
      return route.fulfill({ status: 202, json: lifecycle })
    }
    if (failLifecycleRead) return route.fulfill({ status: 503, json: { detail: 'Controlled refresh failure' } })
    return route.fulfill({ json: lifecycle })
  })
  await page.goto(`${process.env.SETTINGS_TEST_URL || 'http://127.0.0.1:5173'}/settings`)
  const trigger = page.getByRole('button', { name: 'Language model', exact: true })
  const context = page.getByLabel('Max context length (tokens)')
  const menu = page.getByRole('group', { name: 'Language model options' })
  const load = page.getByRole('button', { name: 'Load', exact: true })
  const open = async () => { await trigger.click(); await menu.waitFor() }
  const choose = async name => { await open(); await menu.getByRole('button', { name }).click() }
  const waitFor = async predicate => { for (let n = 0; n < 100 && !predicate(); n++) await page.waitForTimeout(50); assert(predicate()) }
  const finishLoad = async () => {
    await waitFor(() => !!releaseLoad)
    releaseLoad(); releaseLoad = undefined
    await page.getByRole('button', { name: 'Loading…', exact: true }).waitFor()
    await waitFor(() => lifecycle.state === 'loading')
    assert.equal(await page.getByRole('button', { name: 'Loaded', exact: true }).count(), 0)
    lifecycle.state = 'ready'
    await page.getByRole('button', { name: 'Loaded', exact: true }).waitFor()
  }
  await trigger.waitFor()
  for (const name of ['Image generation', 'Video generation', 'Text to speech', 'Speech to text']) assert(await page.getByLabel(name, { exact: true }).isDisabled())
  assert(await page.getByRole('switch', { name: 'Whisper S1 Mini formatting' }).isDisabled())
  assert(await context.isDisabled())
  assert(await load.isDisabled())
  assert.equal(await page.getByRole('button', { name: /Save|Unload|Close model/ }).count(), 0)
  assert.equal(await page.locator('.runtime-panel').count(), 0)
  await open()
  assert.equal(await page.getByRole('dialog').count(), 0)
  assert(await menu.getByRole('button', { name: /Qwen3.6.*not downloaded/ }).isDisabled())
  await page.keyboard.press('End')
  assert(await menu.getByRole('button', { name: 'Download GLM-5.3-Flash-NVFP4', exact: true }).evaluate(element => element === document.activeElement))
  await page.keyboard.press('Escape')
  assert(await trigger.evaluate(element => element === document.activeElement))
  await open()
  await page.getByRole('heading', { name: 'Models', exact: true }).click()
  assert.equal(await menu.count(), 0)
  await open()
  await menu.getByRole('button', { name: 'Download Qwen3.6-35B-A3B-NVFP4', exact: true }).click()
  await page.getByRole('progressbar').waitFor()
  await page.getByRole('button', { name: 'Cancel download', exact: true }).click()
  await page.getByRole('button', { name: 'Retry download', exact: true }).click()
  catalog.models[0].status = 'failed'; catalog.models[0].error = 'Controlled download failure'
  await page.getByRole('alert').filter({ hasText: 'Controlled download failure' }).waitFor()
  await page.getByRole('button', { name: 'Retry download', exact: true }).click()
  catalog.models[0].status = 'complete'
  await page.getByRole('progressbar').waitFor({ state: 'hidden' })
  await choose(/Qwen3.6/)
  assert.equal(catalog.selected_model_id, null)
  assert.equal(await context.inputValue(), '65536')
  assert.deepEqual(await context.locator('option').evaluateAll(options => options.map(option => [option.textContent, option.value])), [
    ['8k', '8192'], ['16k', '16384'], ['32k', '32768'], ['64k', '65536'], ['128k', '131072'], ['500k', '500000'], ['1M', '1000000'], ['Architecture maximum', ''],
  ])
  assert(await context.getByRole('option', { name: '500k', exact: true }).isDisabled())
  await context.selectOption('131072')
  assert.equal(catalog.models[0].context_limit, 65536)
  await load.evaluate(element => { element.click(); element.click() })
  await waitFor(() => !!releaseLoad)
  assert.equal(mutations.filter(([, path]) => path === '/model-lifecycle/load').length, 1)
  assert.deepEqual(mutations.at(-1)[2], { model_id: 'small', context_limit: 131072 })
  await finishLoad()
  assert(await page.getByRole('button', { name: 'Loaded', exact: true }).isDisabled())
  lifecycle.state = 'offloaded'
  await page.waitForTimeout(1700)
  assert(await page.getByRole('button', { name: 'Loaded', exact: true }).isDisabled())
  await page.reload()
  await page.waitForFunction(() => document.querySelector('.model-context select')?.value === '131072')
  // Successful loading clears drafts so subsequent server changes are visible.
  catalog.selected_model_id = 'medium'
  await page.waitForFunction(() => document.querySelector('#model-LLM')?.textContent.includes('gpt-oss'))
  catalog.models[1].context_limit = 4096
  await page.waitForFunction(() => document.querySelector('.model-context select')?.value === '4096')
  assert.equal(await context.locator('option:checked').innerText(), '4096 (saved)')
  await page.reload()
  await page.waitForFunction(() => document.querySelector('.model-context select')?.value === '4096')
  await context.selectOption('65536')
  await context.selectOption('4096')
  catalog.models[1].context_limit = 6000
  await context.getByRole('option', { name: '6000 (saved)', exact: true }).waitFor({ state: 'attached' })
  assert.equal(await context.inputValue(), '4096')
  assert.equal(await context.locator('option:checked').innerText(), '4096 (draft)')
  catalog.models[1].context_limit = 300000
  await page.reload()
  await page.waitForFunction(() => document.querySelector('.model-context select')?.value === '300000')
  assert(await context.getByRole('option', { name: '300000 (saved)', exact: true }).isDisabled())
  assert(await load.isDisabled())
  catalog.models[1].context_limit = null
  catalog.models[1].architecture_context_limit = null
  await page.reload()
  await page.waitForFunction(() => document.querySelector('#context-medium')?.value === '')
  assert.equal(await context.locator('option:checked').innerText(), 'Architecture maximum')
  assert(!(await context.getByRole('option', { name: '1M', exact: true }).isDisabled()))
  failLoad = true
  await load.click()
  await page.getByRole('alert').filter({ hasText: 'Controlled busy load failure' }).waitFor()
  assert.equal(catalog.selected_model_id, 'medium')
  assert.equal(lifecycle.model_id, 'small')
  assert.equal(await page.getByRole('button', { name: 'Loaded', exact: true }).count(), 0)
  failLoad = false
  await load.click()
  await finishLoad()
  assert.deepEqual(mutations.at(-1)[2], { model_id: 'medium', context_limit: null })
  await context.selectOption('1000000')
  assert.equal(catalog.models[1].context_limit, null)
  await load.click()
  await finishLoad()
  assert.equal(catalog.models[1].context_limit, 1000000)
  await page.reload()
  await page.waitForFunction(() => document.querySelector('#context-medium')?.value === '1000000')
  await choose(/Qwen3.6/)
  assert.equal(catalog.selected_model_id, 'medium')
  await load.click()
  await waitFor(() => !!releaseLoad)
  failLifecycleRead = true
  releaseLoad(); releaseLoad = undefined
  await page.getByRole('alert').filter({ hasText: 'Controlled refresh failure' }).first().waitFor()
  await page.waitForTimeout(1700)
  assert(await page.getByRole('button', { name: 'Loading…', exact: true }).isDisabled())
  failLifecycleRead = false
  lifecycle.state = 'ready'
  await page.getByRole('button', { name: 'Loaded', exact: true }).waitFor()
  assert.equal(catalog.selected_model_id, 'small')
  await page.reload()
  await page.getByRole('button', { name: 'Loaded', exact: true }).waitFor()
  assert.doesNotMatch(await page.locator('.settings-layout').innerText(), /Saved:|Blank uses|larger windows|Selection does not|Model card|disk reserve|Load selected model|Runtime lifecycle/)
  await page.screenshot({ path: '/tmp/kadan-settings-load-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 768, height: 1024 })
  assert.equal(await page.locator('.model-llm-details').evaluate(element => getComputedStyle(element).gridColumnStart), '1')
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
  await page.screenshot({ path: '/tmp/kadan-settings-load-tablet.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await open()
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
  await page.screenshot({ path: '/tmp/kadan-settings-load-mobile.png', fullPage: true })
  await page.keyboard.press('End')
  await page.keyboard.press('Tab')
  await menu.waitFor({ state: 'hidden' })
  assert.deepEqual(errors, [])
  console.log('PASS unified settings: one Load, loading/ready/offloaded, duplicate guard, failure/retry, model/context drafts and persistence, download/cancel/retry, keyboard and desktop/tablet/mobile')
} finally {
  await browser.close()
}
