import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE ?? '/opt/codex/runtimes/cua/lib/node_modules/playwright')
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] })
try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    const page = await browser.newPage({ viewport })
    const model = { id: 'h3-fl2va', repo_id: 'MiniMaxAI/MiniMax-H3', revision: 'fixture', license: 'community', estimated_bytes: 144e9, kind: 'video', display_name: 'MiniMax H3 FL2VA', license_url: 'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE', license_notice: 'MiniMax H3’s license excludes use in the US, EU, UK and South Korea, including personal use. Continuing does not grant rights under the license.', inference_available: false, status: 'not_downloaded', downloaded_bytes: 0, total_bytes: 0, error: null, context_limit: null, architecture_context_limit: null }
    const llm = { ...model, id: 'small', kind: 'llm', repo_id: 'nvidia/Qwen', display_name: null, license_notice: null, license_url: null, inference_available: true, context_limit: 65536 }
    const status = () => ({ models: [llm, model], selected_model_id: null })
    let posts = 0, deletes = 0, fail = false
    await page.route('**/model-lifecycle', route => route.fulfill({ json: { state: 'unloaded', model_id: null, error: null } }))
    await page.route('**/v1/models**', async route => {
      const request = route.request()
      if (request.method() === 'POST') {
        posts += 1
        assert.equal(request.postDataJSON().license_acknowledged, true)
        if (fail) return route.fulfill({ status: 503, json: { detail: 'Temporary download failure' } })
        model.status = 'downloading'; model.total_bytes = 144e9; model.downloaded_bytes = 2e9
      }
      if (request.method() === 'DELETE') { deletes += 1; model.status = 'cancelled' }
      await route.fulfill({ json: status() })
    })
    await page.goto(process.env.KADAN_UI_URL ?? 'http://127.0.0.1:15234/settings')
    const open = async () => {
      await page.locator('#model-Video').click()
      await page.getByRole('button', { name: /^(Download|Retry download) MiniMax/ }).click()
      await page.getByRole('dialog').waitFor()
    }
    await open()
    assert.match(await page.getByRole('dialog').innerText(), /including personal use/)
    await page.screenshot({ path: `/tmp/kadan-h3-license-${viewport.width}.png`, fullPage: true })
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()
    assert.equal(posts, 0)
    await open(); await page.keyboard.press('Escape'); assert.equal(posts, 0)
    await open()
    await page.getByRole('button', { name: 'Continue', exact: true }).evaluate(button => { button.click(); button.click() })
    await page.getByRole('button', { name: 'Cancel download', exact: true }).waitFor()
    assert.equal(posts, 1)
    await page.getByRole('button', { name: 'Cancel download', exact: true }).click()
    await page.getByRole('button', { name: 'Retry download', exact: true }).waitFor()
    assert.equal(deletes, 1)
    await page.getByRole('button', { name: 'Retry download', exact: true }).click()
    await page.getByRole('button', { name: 'Cancel', exact: true }).click(); assert.equal(posts, 1)
    fail = true
    await page.getByRole('button', { name: 'Retry download', exact: true }).click()
    await page.getByRole('button', { name: 'Continue', exact: true }).click()
    await page.getByText('Temporary download failure', { exact: true }).waitFor()
    assert.equal(posts, 2)
    fail = false
    await page.getByRole('button', { name: 'Retry download', exact: true }).click()
    await page.getByRole('button', { name: 'Continue', exact: true }).click()
    await page.getByRole('button', { name: 'Cancel download', exact: true }).waitFor()
    assert.equal(posts, 3)
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    await page.screenshot({ path: `/tmp/kadan-h3-progress-${viewport.width}.png`, fullPage: true })
    await page.close()
    console.log(`H3 acknowledgement/cancel/escape/repeated-click/failure/retry/progress passed at ${viewport.width}px`)
  }
} finally { await browser.close() }
