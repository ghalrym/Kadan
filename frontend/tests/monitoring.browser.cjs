// Optional integration smoke: run with a local API + Vite server and Playwright.
// No model/GPU required. Requests deliberately exercise unavailable/invalid chat.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
const assert = require('node:assert/strict')
const base = process.env.MONITORING_TEST_URL || 'http://127.0.0.1:5173'
;(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROMIUM_PATH
      ? { executablePath: process.env.CHROMIUM_PATH }
      : {}),
    args: ['--no-sandbox'],
  })
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    })
    const errors = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.goto(`${base}/requests`)
    await page
      .getByText('No requests on this page.', { exact: false })
      .waitFor()
    assert.equal(
      await page.getByText('Sample metrics', { exact: true }).count(),
      0,
    )
    const unavailable = await page.request.post(`${base}/v1/chat/completions`, {
      data: { messages: [{ role: 'user', text: 'PRIVATE BROWSER PROMPT' }] },
    })
    assert.equal(unavailable.status(), 503)
    for (let i = 0; i < 26; i++) {
      assert.equal(
        (
          await page.request.post(`${base}/v1/chat/completions`, { data: {} })
        ).status(),
        422,
      )
    }
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await page.getByRole('button', { name: 'Next', exact: true }).waitFor()
    await page.waitForFunction(
      () =>
        !Array.from(document.querySelectorAll('button')).find(
          (b) => b.textContent === 'Next',
        ).disabled,
    )
    await page.getByRole('button', { name: 'Next', exact: true }).click()
    await page.getByText('Page 2', { exact: true }).waitFor()
    await page.getByRole('cell', { name: '503', exact: true }).waitFor()
    await page
      .getByRole('row')
      .filter({ has: page.getByRole('cell', { name: '503', exact: true }) })
      .getByRole('link')
      .click()
    await page.getByText('HTTP 503;', { exact: false }).first().waitFor()
    assert.equal(
      (await page.locator('body').innerText()).includes(
        'PRIVATE BROWSER PROMPT',
      ),
      false,
    )
    await page.getByRole('link', { name: 'Close', exact: true }).click()
    await page.getByLabel('HTTP status', { exact: true }).selectOption('503')
    await page.getByText('Page 1', { exact: true }).waitFor()
    await page.getByRole('cell', { name: '503', exact: true }).waitFor()
    assert.equal(
      await page.getByRole('cell', { name: '422', exact: true }).count(),
      0,
    )
    await page
      .getByLabel('Search requests', { exact: true })
      .fill('absent-observation')
    await page
      .getByText('No requests on this page.', { exact: false })
      .waitFor()
    await page.route('**/v1/requests?**', async (route) => {
      const search = new URL(route.request().url()).searchParams.get('search')
      if (search === 'slow') {
        await new Promise((resolve) => setTimeout(resolve, 700))
        return route
          .fulfill({
            json: {
              requests: [],
              total: 999,
              retention_limit: 1000,
              started_at: '2026-01-01',
            },
          })
          .catch(() => {})
      }
      return route.fulfill({
        status: 503,
        json: { detail: 'Controlled monitoring outage' },
      })
    })
    await page.getByLabel('Search requests', { exact: true }).fill('slow')
    await page.waitForTimeout(100)
    await page.getByLabel('Search requests', { exact: true }).fill('failed')
    await page
      .getByRole('alert')
      .filter({ hasText: 'Monitoring API returned HTTP 503' })
      .waitFor()
    await page.waitForTimeout(800)
    assert.equal(
      (await page.locator('body').innerText()).includes('999 matching'),
      false,
    )
    assert.equal(await page.locator('.request-table tbody tr').count(), 0)
    await page.goto(`${base}/requests/expired-id`)
    await page
      .getByRole('alert')
      .filter({ hasText: 'Record not found' })
      .waitFor()
    assert.deepEqual(errors, [])
    console.log(
      'Monitoring browser smoke passed: empty, real 503/422, paging/filter/search/detail, privacy, error and stale-response suppression.',
    )
  } finally {
    await browser.close()
  }
})().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
