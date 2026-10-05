/** Render monitoring with controlled observations; never writes backend data.
 * Set DESIGN_TEST_URL, PLAYWRIGHT_MODULE and PLAYWRIGHT_CHROMIUM_PATH as needed.
 */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdir } from 'node:fs/promises'
const { chromium } = createRequire(import.meta.url)(
  process.env.PLAYWRIGHT_MODULE || 'playwright',
)
const browser = await chromium.launch({
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH,
  headless: true,
  args: ['--no-sandbox'],
})
const root = process.env.DESIGN_TEST_URL || 'http://127.0.0.1:15231'
const screenshots = '/tmp/kadan-review32-screenshots'
await mkdir(screenshots, { recursive: true })
try {
  const page = await browser.newPage({
    viewport: { width: 2488, height: 1171 },
  })
  const errors = []
  page.on('pageerror', (error) => errors.push(error.message))
  const records = ['LLM', 'Image', 'Video', 'TTS', 'STT', 'Decision'].map(
    (type, i) => ({
      id: `observed-${i}`,
      time: '2026-10-05T01:00:00Z',
      type,
      model: type === 'LLM' ? 'Observed model' : null,
      status: i === 1 ? 503 : 200,
      latency: '0.125 s',
      latency_ms: 125,
      endpoint: '/v1/example',
      prompt: 'Request: 24 bytes',
      output: i === 1 ? 'HTTP 503' : 'Response: 12 bytes',
      request_bytes: 24,
      response_bytes: 12,
    }),
  )
  let failDetail = false,
    total = 6
  await page.route('**/v1/**', (route) => {
    const url = new URL(route.request().url())
    if (url.pathname === '/v1/metrics')
      return route.fulfill({
        json: {
          requests_per_minute: 6,
          p50_latency_seconds: 0.125,
          error_rate_percent: 16.7,
          active_requests: 0,
          window_truncated: false,
          resources: [{ label: 'Host RAM', used: 12, total: 64 }],
          resource_errors: [],
        },
      })
    if (url.pathname === '/v1/requests') {
      const type = url.searchParams.get('type')
      const requests = type ? records.filter((r) => r.type === type) : records
      return route.fulfill({
        json: {
          requests,
          total: type ? requests.length : total,
          retention_limit: 500,
          started_at: records[0].time,
        },
      })
    }
    if (failDetail) return route.fulfill({ status: 503, json: {} })
    return route.fulfill({
      json: { request: records.find((r) => url.pathname.endsWith(r.id)) },
    })
  })
  await page.goto(root + '/requests')
  await page.getByRole('link', { name: 'View request observed-0' }).waitFor()
  assert.equal(await page.locator('tbody tr').count(), 6)
  assert.equal(
    await page.getByRole('combobox', { name: 'HTTP status' }).count(),
    0,
  )
  assert.equal(
    await page.getByRole('button', { name: 'Next', exact: true }).count(),
    0,
  )
  const filters = await page.locator('.filter-chip').first().boundingBox(),
    search = await page
      .getByRole('textbox', { name: 'Search requests' })
      .boundingBox()
  assert(
    Math.abs(filters.y - search.y) < 8,
    'Desktop filters/search must share the original horizontal row',
  )
  assert.equal(
    await page
      .locator('.stat-value > span')
      .allTextContents()
      .then((v) => v.join('')),
    's%',
  )
  await page.screenshot({ path: `${screenshots}/desktop-requests.png` })
  await page.getByRole('link', { name: 'View request observed-1' }).click()
  await page.locator('.drawer-header .status-badge').waitFor()
  assert.equal(
    await page.locator('.drawer-header .status-badge').textContent(),
    '503',
  )
  assert(await page.locator('.drawer-header .status-badge').evaluate(e => e.classList.contains('error')))
  assert.equal(await page.locator('.request-drawer .error-panel').count(), 1)
  assert(await page.getByRole('button', { name: 'Copy as cURL' }).isDisabled())
  assert(
    await page.getByRole('button', { name: 'Replay request' }).isDisabled(),
  )
  assert(await page.getByRole('button', { name: 'Copy ID' }).isEnabled())
  await page.getByText('Advanced', { exact: true }).click()
  await page.screenshot({ path: `${screenshots}/desktop-drawer.png` })
  await page.getByRole('link', { name: 'Close', exact: true }).click()
  total = 30
  await page.getByRole('button', { name: 'Refresh requests' }).click()
  await page.getByRole('button', { name: 'Next', exact: true }).waitFor()
  assert(
    await page.getByRole('button', { name: 'Next', exact: true }).isEnabled(),
  )
  total = 6
  await page.setViewportSize({ width: 390, height: 1000 })
  await page.goto(root + '/requests')
  await page.getByRole('link', { name: 'View request observed-0' }).waitFor()
  await page.screenshot({ path: `${screenshots}/mobile-requests.png` })
  await page.getByRole('link', { name: 'View request observed-0' }).click()
  await page.locator('.drawer-header .status-badge').waitFor()
  await page.screenshot({ path: `${screenshots}/mobile-drawer.png` })
  assert(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  )
  failDetail = true
  await page.reload()
  await page.getByRole('heading', { name: 'Request unavailable' }).waitFor()
  failDetail = false
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await page.locator('.drawer-header .status-badge').waitFor()
  assert.deepEqual(errors, [])
  console.log(
    'PASS: populated desktop/mobile table, horizontal filter/search layout, stat units, conditional pagination, original drawer header/footer, error and retry; four screenshots',
  )
} finally {
  await browser.close()
}
