const {
  chromium,
} = require('/opt/codex/runtimes/cua/lib/node_modules/playwright')
;(async () => {
  const browser = await chromium.launch({
    executablePath: '/usr/bin/chromium',
    headless: true,
    args: ['--no-sandbox'],
  })
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } })
    const posted = []
    await page.route('**/v1/audio/speech**', async (route) => {
      const req = route.request()
      if (req.url().endsWith('/models'))
        return route.fulfill({
          json: [
            {
              id: 'qwen-tts-1.7b-design',
              name: 'Qwen3-TTS 1.7B VoiceDesign',
              mode: 'describe',
            },
            {
              id: 'qwen-tts-1.7b-base',
              name: 'Qwen3-TTS 1.7B Base',
              mode: 'clone',
            },
            {
              id: 'qwen-tts-0.6b-custom',
              name: 'Qwen3-TTS 0.6B CustomVoice',
              mode: 'custom',
            },
          ],
        })
      if (req.method() === 'POST') {
        posted.push(req.postDataJSON())
        return route.fulfill({
          json: {
            audio: {
              voice: 'Qwen3-TTS',
              meta: 'WAV',
              script: 'hello',
              time: '1s',
              audio_base64: 'UklGRg==',
              mime_type: 'audio/wav',
            },
          },
        })
      }
      return route.fulfill({
        json: { audio: [], voice_description: '', script: '' },
      })
    })
    await page.goto(
      `${process.env.KADAN_BROWSER_URL || 'http://localhost:15236'}/tts`,
    )
    await page.locator('#speech-model').selectOption('qwen-tts-1.7b-design')
    await page.locator('#speech-description').fill('warm')
    await page.locator('#speech-script').fill('hello')
    await page
      .getByRole('button', { name: 'Generate speech', exact: true })
      .click()
    await page.locator('audio').waitFor()
    if (posted.length !== 1) throw Error('duplicate')
    await page.screenshot({ path: `/tmp/qwen-${width}.png` })
    await page.goto(
      `${process.env.KADAN_BROWSER_URL || 'http://localhost:15236'}/tts/clone`,
    )
    await page.locator('#speech-model').selectOption('qwen-tts-1.7b-base')
    await page
      .locator('#speech-sample')
      .setInputFiles({
        name: 'sample.wav',
        mimeType: 'audio/wav',
        buffer: Buffer.from('RIFF'),
      })
    await page.locator('#speech-reference').fill('reference')
    await page.locator('#speech-script').fill('hello')
    await page
      .getByRole('button', { name: 'Generate speech', exact: true })
      .click()
    await page.locator('audio').waitFor()
    if (
      posted[1].voice.sample !== 'UklGRg==' ||
      posted[1].voice.transcript !== 'reference'
    )
      throw Error('clone payload')
    if (
      await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth,
      )
    )
      throw Error('horizontal overflow')
    await page.close()
  }
  await browser.close()
  console.log('Desktop/mobile describe + clone upload/playback passed')
})().catch((e) => {
  console.error(e)
  process.exit(1)
})
