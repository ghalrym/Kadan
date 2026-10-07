import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE ?? '/opt/codex/runtimes/cua/lib/node_modules/playwright')
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true, args: ['--no-sandbox', '--disable-gpu'] })
try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    const page = await browser.newPage({ viewport })
    const model = { id: 'qwen-image-2.1', repo_id: 'Qwen/Qwen-Image-2.1', revision: 'fixture', license: 'community', estimated_bytes: 33.1e9, kind: 'image', display_name: 'Qwen-Image-2.1', license_url: 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE', license_notice: 'Qwen-Image-2.1 is licensed for research and evaluation only. Commercial use requires a separate license; continuing does not grant commercial rights.', inference_available: false, status: 'not_downloaded', downloaded_bytes: 0, total_bytes: 0, error: null, context_limit: null, architecture_context_limit: null }
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
        model.status = 'downloading'; model.total_bytes = 33.1e9; model.downloaded_bytes = 2e9
      }
      if (request.method() === 'DELETE') { deletes += 1; model.status = 'cancelled' }
      await route.fulfill({ json: status() })
    })
    await page.goto(process.env.KADAN_UI_URL ?? 'http://127.0.0.1:15238/settings')
    const open = async () => {
      await page.locator('#model-Image').click()
      await page.getByRole('button', { name: /^(Download|Retry download) Qwen/ }).click()
      await page.getByRole('dialog').waitFor()
    }
    await open()
    assert.match(await page.getByRole('dialog').innerText(), /research and evaluation only/)
    await page.screenshot({ path: `/tmp/kadan-qwen-image-license-${viewport.width}.png`, fullPage: true })
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
    await page.screenshot({ path: `/tmp/kadan-qwen-image-progress-${viewport.width}.png`, fullPage: true })
    await page.close()
    console.log(`Qwen Image acknowledgement/cancel/escape/repeated-click/failure/retry/progress passed at ${viewport.width}px`)
  }
} finally { await browser.close() }
