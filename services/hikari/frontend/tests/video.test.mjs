import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'

async function videoApi(
  settings = { video_defaults: { duration: 5, resolution: '480p' } },
) {
  const calls = []
  const job = { id: 'job', prompt: 'scene', status: 'Queued', progress: 0 }
  const sdk = new SyntheticModule(
    ['cancelVideo', 'generateVideo', 'getSettings', 'getVideo', 'listVideos'],
    function () {
      for (const name of [
        'cancelVideo',
        'generateVideo',
        'getSettings',
        'getVideo',
        'listVideos',
      ])
        this.setExport(name, async (args) => {
          calls.push(args)
          return {
            response: { ok: true },
            data: name === 'getSettings' ? settings : { job },
          }
        })
    },
  )
  const source = await readFile(
    new URL('../src/api/video.ts', import.meta.url),
    'utf8',
  )
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link((specifier) => {
    assert.equal(specifier, './generated/sdk.gen')
    return sdk
  })
  await module.evaluate()
  return { api: module.namespace, calls }
}

test('submits 1–3 second clips unchanged and keeps accepted jobs pending', async () => {
  const { api, calls } = await videoApi()
  for (const seconds of [1, 2, 3]) {
    const job = await api.submitVideo(
      { prompt: 'scene', seconds, fps: 24 },
      new AbortController().signal,
    )
    assert.equal(calls.at(-1).body.seconds, seconds)
    assert.equal(api.isPendingVideo(job), true)
  }
})

test('rejects unsupported durations before contacting the API', async () => {
  const { api, calls } = await videoApi()
  for (const seconds of [0, 1.5, 16])
    await assert.rejects(
      api.submitVideo(
        { prompt: 'scene', seconds, fps: 24 },
        new AbortController().signal,
      ),
      /1 to 15 seconds/,
    )
  assert.equal(calls.length, 0)
})

test('reads fixed defaults from the API without a frontend hardware policy', async () => {
  const { api } = await videoApi({
    video_defaults: { duration: 5, resolution: '480p' },
  })
  assert.deepEqual(await api.loadVideoDefaults(new AbortController().signal), {
    duration: 5,
    resolution: '480p',
  })
})

test('missing video defaults fail without inventing a tier or replacement', async () => {
  const { api } = await videoApi({})
  await assert.rejects(
    api.loadVideoDefaults(new AbortController().signal),
    /Video defaults are unavailable/,
  )
})

test('leaves omitted clip settings for the API defaults and accepts full explicit choices', async () => {
  const { api, calls } = await videoApi()
  await api.submitVideo({ prompt: 'scene' }, new AbortController().signal)
  assert.deepEqual(calls.at(-1).body, { prompt: 'scene' })
  await api.submitVideo(
    { prompt: 'scene', seconds: 15, resolution: '768p', fps: 24 },
    new AbortController().signal,
  )
  assert.equal(calls.at(-1).body.seconds, 15)
  assert.equal(calls.at(-1).body.resolution, '768p')
})
