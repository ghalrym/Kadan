/** Render restored controls against controlled HTTP responses. No backend writes.
 * Run with DESIGN_TEST_URL, PLAYWRIGHT_MODULE and PLAYWRIGHT_CHROMIUM_PATH.
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
const screenshots =
  process.env.DESIGN_SCREENSHOTS || '/tmp/kadan-design-screenshots'
await mkdir(screenshots, { recursive: true })
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  })
  const errors = [],
    posts = [],
    queries = []
  page.on('pageerror', (e) => errors.push(e.message))
  let failHistory = false,
    failSchema = false
  await page.route('**/v1/**', async (route) => {
    const req = route.request(),
      url = new URL(req.url()),
      path = url.pathname
    if (req.method() === 'POST') {
      posts.push({ path, body: req.postDataJSON() })
      return route.fulfill({
        status: 503,
        json: { detail: 'Provider unavailable' },
      })
    }
    let json = {}
    if (path === '/v1/metrics')
      json = {
        requests_per_minute: 0,
        p50_latency_seconds: null,
        error_rate_percent: null,
        active_requests: 0,
        resources: [],
        resource_errors: [],
        window_truncated: false,
      }
    if (path === '/v1/images') {
      if (failHistory) return route.fulfill({ status: 503, json: {} })
      json = { images: [] }
    }
    if (path === '/v1/videos') json = { jobs: [] }
    if (path === '/v1/audio/speech') json = { audio: [] }
    if (path === '/v1/requests') {
      queries.push(url.searchParams.toString())
      json = {
        requests: [],
        total: 0,
        retention_limit: 500,
        started_at: '2026-10-05T00:00:00Z',
      }
    }
    await route.fulfill({ json })
  })
  await page.route('**/openapi.json', (route) =>
    route.fulfill({
      json: failSchema
        ? { paths: null }
        : {
            info: { title: 'Kadan', version: '1' },
            paths: { '/v1/images': { get: { summary: 'List images' } } },
          },
    }),
  )
  await page.goto(root + '/image')
  await page.getByLabel('Empty image gallery').waitFor()
  assert.equal(await page.locator('.image-grid .media-placeholder').count(), 4)
  assert.equal(
    await page
      .getByRole('group', { name: 'Images', exact: true })
      .getByRole('button', { name: '4', exact: true })
      .getAttribute('aria-pressed'),
    'true',
  )
  await page
    .getByRole('textbox', { name: 'Prompt', exact: true })
    .fill('A sunrise')
  const countTwo = page
    .getByRole('group', { name: 'Images', exact: true })
    .getByRole('button', { name: '2', exact: true })
  await countTwo.focus()
  await page.keyboard.press('Space')
  await page
    .getByRole('group', { name: 'Aspect', exact: true })
    .getByRole('button', { name: '16:9', exact: true })
    .click()
  await page.getByRole('button', { name: 'Generate 2 images' }).click()
  await page
    .getByRole('alert')
    .filter({ hasText: 'No image provider' })
    .waitFor()
  assert.equal(posts.at(-1).body.count, 2)
  assert.equal(posts.at(-1).body.aspect, '16:9')
  assert.equal(await page.locator('.image-grid .media-placeholder').count(), 4)
  failHistory = true
  await page.reload()
  await page
    .getByRole('alert')
    .filter({ hasText: 'Could not load image history' })
    .waitFor()
  failHistory = false
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await page.getByLabel('Empty image gallery').waitFor()
  await page.goto(root + '/image/edit')
  assert(await page.getByRole('button', { name: /Drop an image/ }).isDisabled())
  await page
    .getByRole('textbox', { name: 'Describe the edit' })
    .fill('New background')
  assert(await page.getByRole('button', { name: 'Apply edit' }).isDisabled())
  await page.goto(root + '/video')
  await page.getByText('No video jobs.', { exact: true }).waitFor()
  assert.equal(await page.getByRole('alert').count(), 0)
  await page
    .getByRole('textbox', { name: 'Prompt', exact: true })
    .fill('A sunrise')
  await page
    .getByRole('group', { name: 'Resolution' })
    .getByRole('button', { name: '1080p' })
    .click()
  await page.getByRole('button', { name: 'Queue video' }).click()
  await page
    .getByRole('alert')
    .filter({ hasText: 'No job was queued' })
    .waitFor()
  assert.equal(posts.at(-1).body.duration, 8)
  assert.equal(posts.at(-1).body.fps, 24)
  assert.equal(posts.at(-1).body.resolution, '1080p')
  assert.equal(await page.locator('.video-job-card').count(), 0)
  await page.goto(root + '/tts')
  await page.getByText('No generated audio', { exact: true }).waitFor()
  assert(
    (
      await page
        .getByRole('textbox', { name: 'Script', exact: true })
        .inputValue()
    ).length > 0,
  )
  assert(await page.getByRole('button', { name: 'Play audio' }).isDisabled())
  await page.getByRole('button', { name: 'Generate speech' }).click()
  await page
    .getByRole('alert')
    .filter({ hasText: 'No speech provider' })
    .waitFor()
  assert.equal(posts.at(-1).path, '/v1/audio/speech')
  await page.goto(root + '/tts/clone')
  assert(
    await page
      .getByRole('button', { name: /Upload a voice sample/ })
      .isDisabled(),
  )
  assert(
    await page.getByRole('button', { name: 'Generate speech' }).isDisabled(),
  )
  await page.goto(root + '/stt')
  assert(
    await page.getByRole('button', { name: 'Start recording' }).isDisabled(),
  )
  assert(
    await page
      .getByRole('button', { name: 'Submit', exact: true })
      .isDisabled(),
  )
  assert.equal(await page.getByRole('textbox').count(), 0)
  await page.goto(root + '/requests')
  await page.getByText('No requests on this page.', { exact: true }).waitFor()
  assert.deepEqual(await page.locator('thead th').allTextContents(), [
    'Time',
    'Type',
    'Model',
    'TTFT',
    'Tok/s',
    'Status',
    'Latency',
  ])
  await page.getByRole('button', { name: /^Image\b/ }).click()
  await page.waitForResponse(
    (r) => r.url().includes('/v1/requests?') && r.url().includes('type=Image'),
  )
  assert(queries.some((q) => q.includes('type=Image')))
  failSchema = true
  await page.goto(root + '/api')
  await page.getByRole('alert').waitFor()
  failSchema = false
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await page.getByRole('cell', { name: 'List images' }).waitFor()
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 })
    for (const path of [
      '/image',
      '/image/edit',
      '/video',
      '/tts',
      '/tts/clone',
      '/stt',
      '/requests',
      '/api',
    ]) {
      await page.goto(root + path)
      await page.locator('h1').waitFor()
      await page.screenshot({
        path: `${screenshots}/${width}-${path.slice(1).replaceAll('/', '-')}.png`,
        fullPage: true,
      })
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
        `Document overflows: ${width} ${path}`,
      )
    }
  }
  assert.deepEqual(errors, [])
  console.log(
    'PASS: desktop/mobile, keyboard segments, API payloads, 503 failures, history/schema retries, original disabled media controls, monitoring filters; 16 screenshots',
  )
} finally {
  await browser.close()
}
