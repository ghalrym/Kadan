import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'

const checkpoint = (id, overrides = {}) => ({
  id,
  status: 'not_downloaded',
  total_bytes: 0,
  ...overrides,
})
async function settingsApi(
  response = { ok: true },
  data = { models: [], defaults: {} },
  error,
) {
  const calls = []
  const names = [
    'getSettings',
    'downloadModelV1ModelsModelIdDownloadPost',
    'cancelDownloadV1ModelsModelIdDownloadDelete',
    'configureContextV1ModelsModelIdContextPut',
  ]
  const sdk = new SyntheticModule(names, function () {
    for (const name of names)
      this.setExport(name, async (args) => {
        calls.push({ name, args })
        return { response, data, error }
      })
  })
  const source = await readFile(
    new URL('../src/api/settings.ts', import.meta.url),
    'utf8',
  )
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link((name) => {
    assert.equal(name, './generated/sdk.gen')
    return sdk
  })
  await module.evaluate()
  return { api: module.namespace, calls }
}

test('fixed assignments dedupe shared models and retain missing worker rows', async () => {
  const { api } = await settingsApi()
  const model = checkpoint('audio')
  const rows = api.configuredModels({
    defaults: { speech: 'audio', transcription: 'audio', image: 'image' },
    models: [model],
  })
  assert.equal(rows.length, 2)
  assert.equal(rows[0].checkpoint, model)
  assert.deepEqual(rows[0].uses, ['Speech and calls', 'Transcription'])
  assert.equal(rows[1].id, 'image')
  assert.equal(rows[1].checkpoint, null)
})

test('download states distinguish waiting, progress, completion and cancellation', async () => {
  const { api } = await settingsApi()
  assert.equal(api.downloadState(null), 'unassigned')
  assert.equal(api.downloadState(checkpoint('a')), 'not_downloaded')
  assert.equal(
    api.downloadState(checkpoint('a', { status: 'downloading' })),
    'queued',
  )
  assert.equal(
    api.downloadState(
      checkpoint('a', { status: 'downloading', total_bytes: 12 }),
    ),
    'downloading',
  )
  assert.equal(
    api.downloadState(checkpoint('a', { status: 'complete' })),
    'complete',
  )
  assert.equal(
    api.downloadState(checkpoint('a', { status: 'cancelled' })),
    'failed',
  )
})

test('download requires explicit acknowledgement and selects exactly one checkpoint', async () => {
  const models = [checkpoint('image'), checkpoint('text')]
  const { api, calls } = await settingsApi({ ok: true }, { models })
  const signal = new AbortController().signal
  await assert.rejects(api.downloadModel('image', false, signal), /acknowledge/)
  assert.equal(calls.length, 0)
  assert.equal(await api.downloadModel('image', true, signal), models)
  assert.deepEqual(calls[0].args, {
    path: { model_id: 'image' },
    body: { license_acknowledged: true },
    signal,
  })
})

test('cancel targets the requested model and returns observed status', async () => {
  const models = [checkpoint('image', { status: 'cancelling' })]
  const { api, calls } = await settingsApi({ ok: true }, { models })
  const signal = new AbortController().signal
  assert.equal(await api.cancelModelDownload('image', signal), models)
  assert.equal(calls[0].name, 'cancelDownloadV1ModelsModelIdDownloadDelete')
  assert.deepEqual(calls[0].args.path, { model_id: 'image' })
})

test('context rejects invalid limits locally and preserves explicit maximum selection', async () => {
  const { api, calls } = await settingsApi()
  const signal = new AbortController().signal
  for (const value of [0, -1, 1.5, NaN])
    await assert.rejects(
      api.configureModelContext('text', value, signal),
      /positive whole-token/,
    )
  assert.equal(calls.length, 0)
  await api.configureModelContext('text', null, signal)
  assert.deepEqual(calls[0].args.body, { context_limit: null })
  await api.configureModelContext('text', 8192, signal)
  assert.deepEqual(calls[1].args.body, { context_limit: 8192 })
})

test('API failures preserve server detail and never fabricate settings', async () => {
  const bad = await settingsApi({ ok: false, status: 422 }, null, {
    detail: 'Checkpoint unavailable',
  })
  await assert.rejects(
    bad.api.fetchSettings(new AbortController().signal),
    /Checkpoint unavailable/,
  )
  const offline = await settingsApi(null, null)
  await assert.rejects(
    offline.api.fetchSettings(new AbortController().signal),
    /Cannot reach/,
  )
})
