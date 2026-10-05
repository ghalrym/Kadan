const assert = require('node:assert/strict')
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/opt/codex/runtimes/cua/lib/node_modules/playwright')
;(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] })
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } })
      let state = { current: 'a'.repeat(40), available: 'b'.repeat(40), configured: true, authorized: false,
        phase: 'idle', message: 'Update available.', can_cancel: false }
      const actions = []
      let disconnected = false
      await page.route('**/v1/models**', route => route.fulfill({ json: { models: [], selected_model_id: null } }))
      await page.route('**/model-lifecycle', route => route.fulfill({ json: { state: 'unloaded', model_id: null } }))
      await page.route('**/v1/updates**', async route => {
        const request = route.request()
        if (request.method() === 'GET' && disconnected) return route.fulfill({ status: 502, body: 'Restarting' })
        if (request.method() === 'POST') {
          actions.push(request.url().split('/').pop())
          assert.equal(request.headers()['x-kadan-update'], '1')
          if (request.url().endsWith('/pair')) state.authorized = true
          if (request.url().endsWith('/install')) {
            assert.deepEqual(request.postDataJSON(), { commit: 'b'.repeat(40) })
            state = { ...state, phase: 'pulling', message: 'Downloading both images while Kadan keeps running…', can_cancel: true }
          }
          if (request.url().endsWith('/cancel')) state = { ...state, phase: 'failed', message: 'Update cancelled.', can_cancel: false }
        }
        await route.fulfill({ json: state })
      })
      await page.goto((process.env.KADAN_BROWSER_URL || 'http://127.0.0.1:15238') + '/settings')
      await page.getByText('Update available.', { exact: true }).waitFor()
      assert.equal(actions.length, 0)
      assert.equal(await page.getByRole('button', { name: 'Update', exact: true }).isDisabled(), true)
      await page.getByLabel('Authorize updates for this browser').fill('test-only-code-'.repeat(3))
      await page.getByRole('button', { name: 'Authorize', exact: true }).click()
      await page.getByRole('button', { name: 'Update', exact: true }).click()
      await page.getByText('Downloading both images while Kadan keeps running…', { exact: true }).waitFor()
      await page.getByRole('button', { name: 'Cancel update', exact: true }).click()
      await page.getByText('Update cancelled.', { exact: true }).waitFor()
      assert.deepEqual(actions, ['pair', 'install', 'cancel'])
      await page.getByRole('button', { name: 'Update', exact: true }).click()
      disconnected = true
      await page.getByRole('alert').filter({ hasText: 'Reconnecting to Kadan' }).waitFor()
      const reloaded = page.waitForEvent('framenavigated', frame => frame === page.mainFrame())
      state = { ...state, current: 'b'.repeat(40), available: null, phase: 'complete', message: 'Update complete.', can_cancel: false }
      disconnected = false
      await reloaded
      await page.getByText('Kadan ' + 'b'.repeat(12), { exact: true }).waitFor()
      assert.deepEqual(actions, ['pair', 'install', 'cancel', 'install'])
      state = { ...state, phase: 'recovery_required', error: 'No forced container kill or database downgrade was attempted.' }
      await page.getByRole('alert').filter({ hasText: 'No forced container kill' }).waitFor()
      assert.equal(await page.getByRole('button', { name: 'Update', exact: true }).isDisabled(), true)
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
      await page.screenshot({ path: `/tmp/kadan-updates-${width}.png`, fullPage: true })
      console.log(`Owner authorization, deliberate install/cancel, reconnect/reload, recovery error and layout passed at ${width}px`)
      await page.close()
    }
  } finally { await browser.close() }
})().catch(error => { console.error(error); process.exitCode = 1 })
