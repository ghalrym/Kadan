const { chromium } = require('/opt/codex/runtimes/cua/lib/node_modules/playwright')
const assert = require('node:assert/strict')
;(async () => {
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', args: ['--no-sandbox'] })
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } })
    const uploads = [], jobs = []
    let failUpload = true
    await page.route('**/v1/videos**', route => {
      const request = route.request()
      if (request.url().includes('/inputs?')) {
        uploads.push({ url: request.url(), type: request.headers()['content-type'], body: request.postDataBuffer() })
        if (failUpload) return route.fulfill({ status: 503, json: { detail: 'fixture failure' } })
        return route.fulfill({ json: { id: request.url().includes('kind=image') ? 'a'.repeat(32) : 'b'.repeat(32) } })
      }
      if (request.url().includes('/generations')) {
        jobs.push(request.postDataJSON())
        return route.fulfill({ json: { job: { id: 'test', prompt: 'Singing', duration: '8s', resolution: '720p', aspect: 'wide', fps: '16', progress: 0, time: '', status: 'Queued', thumbnail: '', progressText: 'Queued' } } })
      }
      return route.fulfill({ json: { jobs: [] } })
    })
    await page.goto(`${process.env.KADAN_BROWSER_URL || 'http://localhost:15250'}/video`)
    await page.getByLabel('Model', { exact: true }).selectOption('wan22-s2v-14b')
    await page.getByLabel('Prompt', { exact: true }).fill('Singing')
    assert(await page.getByRole('button', { name: 'Queue video' }).isDisabled())
    await page.getByLabel('Reference image').setInputFiles({ name: 'ref.png', mimeType: 'image/png', buffer: Buffer.from('image fixture') })
    await page.getByLabel('Speech audio (WAV)').setInputFiles({ name: 'speech.wav', mimeType: 'audio/wav', buffer: Buffer.from('RIFF fixture') })
    await page.getByRole('button', { name: 'Queue video' }).click()
    await page.getByRole('alert').waitFor()
    assert.equal(jobs.length, 0)
    failUpload = false
    await page.getByRole('button', { name: 'Queue video' }).click()
    await page.getByText('1 pending · 0 done').waitFor()
    assert.equal(jobs.length, 1)
    assert.equal(jobs[0].model, 'wan22-s2v-14b')
    assert.equal(jobs[0].image_id, 'a'.repeat(32))
    assert.equal(jobs[0].audio_id, 'b'.repeat(32))
    assert.equal(jobs[0].fps, 16)
    assert.equal(uploads.at(-1).type, 'audio/wav')
    assert.equal(uploads.at(-1).body.toString(), 'RIFF fixture')
    await page.screenshot({ path: `/tmp/wan-s2v-${width}.png`, fullPage: true })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await page.close()
  }
  await browser.close()
  console.log('S2V desktop/mobile uploads, validation, failure/retry and native request passed')
})().catch(error => { console.error(error); process.exit(1) })
