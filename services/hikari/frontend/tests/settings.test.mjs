import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'

function checkpoint(id, overrides = {}) {
  return {
    id, display_name: id, status: 'not_downloaded', downloaded_bytes: 0, total_bytes: 0,
    estimated_bytes: 100, license_notice: null, license_url: null, error: null, ...overrides,
  }
}

async function load(path) {
  return new SourceTextModule(stripTypeScriptTypes(await readFile(new URL(path, import.meta.url), 'utf8')))
}

async function settingsApi(response = { ok: true }, data = { models: [] }, error = undefined) {
  const calls = []
  const names = ['listModelsV1ModelsGet', 'downloadModelV1ModelsModelIdDownloadPost']
  const sdk = new SyntheticModule(names, function () {
    for (const name of names)
      this.setExport(name, async args => { calls.push({ name, args }); return { response, data, error } })
  })
  const models = await load('../src/data/models.ts')
  await models.link(() => assert.fail('models.ts has no imports'))
  const module = await load('../src/api/settings.ts')
  await module.link(specifier => ({ './generated/sdk.gen': sdk, '../data/models': models })[specifier])
  await module.evaluate()
  const capabilities = models.namespace.capabilities
  const rows = overrides => capabilities.map(capability => ({
    capability, model: capability.id in overrides ? overrides[capability.id] : checkpoint(capability.id, { status: 'complete' }),
  }))
  return { api: module.namespace, calls, capabilities, rows }
}

test('every capability reads its fixed checkpoint from the model snapshot', async () => {
  const { api, calls, capabilities } = await settingsApi({ ok: true }, { models: [
    checkpoint('qwen3.6-35b-a3b-q4-k-m'), checkpoint('lfm3-vl-2b-bf16'), checkpoint('lfm3-1.7b-bf16'),
  ] })
  const result = await api.fetchModels(new AbortController().signal)
  assert.deepEqual(result.map(row => row.capability.label),
    ['Language', 'Vision', 'Image', 'Video', 'Text to speech', 'Speech to text', 'Speech formatter', 'Decisions'])
  assert.equal(result[0].model.id, 'lfm3-1.7b-bf16')
  assert.equal(result[1].model.id, 'lfm3-vl-2b-bf16')
  assert.equal(result[2].model, null)
  assert.equal(capabilities.length, 8)
  assert.deepEqual(calls.map(call => call.name), ['listModelsV1ModelsGet'])
})

test('download states distinguish queued, active, finished and failed checkpoints', async () => {
  const { api } = await settingsApi()
  assert.equal(api.downloadState(null), 'unavailable')
  assert.equal(api.downloadState(checkpoint('a')), 'not_downloaded')
  assert.equal(api.downloadState(checkpoint('a', { status: 'downloading' })), 'queued')
  assert.equal(api.downloadState(checkpoint('a', { status: 'downloading', total_bytes: 10 })), 'downloading')
  assert.equal(api.downloadState(checkpoint('a', { status: 'complete' })), 'complete')
  for (const status of ['failed', 'cancelled'])
    assert.equal(api.downloadState(checkpoint('a', { status })), 'failed')
})

test('overall progress weights each checkpoint by size', async () => {
  const { api, rows } = await settingsApi()
  const language = checkpoint('lfm3-1.7b-bf16', { status: 'downloading', downloaded_bytes: 150, total_bytes: 300 })
  const active = rows({ 'lfm3-1.7b-bf16': language, 'lfm3-vl-2b-bf16': null })
  assert.equal(api.modelsProgress(active), 83)
  assert.equal(api.modelsDownloading(active), true)
  assert.equal(api.modelsComplete(active), false)
  assert.equal(api.modelsComplete(rows({})), true)
  assert.equal(api.modelsComplete(rows({ 'whisper-medium': null })), false)
})

test('license notices are shown only for checkpoints still to download', async () => {
  const { api, rows } = await settingsApi()
  const notice = { license_notice: 'LFM Open License', license_url: 'https://example.invalid' }
  const notices = api.licenseNotices(rows({
    'lfm3-1.7b-bf16': checkpoint('lfm3-1.7b-bf16', notice),
    'lfm3-vl-2b-bf16': checkpoint('lfm3-vl-2b-bf16', { ...notice, status: 'complete' }),
  }))
  assert.deepEqual(notices.map(model => model.id), ['lfm3-1.7b-bf16'])
})

test('downloads are offered for missing, failed or cancelled checkpoints but not active ones', async () => {
  const { api, rows } = await settingsApi()
  for (const status of ['not_downloaded', 'failed', 'cancelled'])
    assert.equal(api.modelsCanDownload(rows({ 'whisper-medium': checkpoint('whisper-medium', { status }) })), true)
  for (const status of ['downloading', 'cancelling', 'complete'])
    assert.equal(api.modelsCanDownload(rows({ 'whisper-medium': checkpoint('whisper-medium', { status }) })), false)
})

test('downloading requires acknowledgement and queues only idle incomplete checkpoints', async () => {
  const { api, calls, rows } = await settingsApi({ ok: true }, { models: [] })
  const signal = new AbortController().signal
  const pending = rows({
    'lfm3-1.7b-bf16': checkpoint('lfm3-1.7b-bf16'),
    'lfm3-vl-2b-bf16': checkpoint('lfm3-vl-2b-bf16', { status: 'downloading' }),
    'whisper-medium': checkpoint('whisper-medium', { status: 'failed' }),
  })
  await assert.rejects(api.downloadModels(pending, false, signal), /acknowledge the model licenses/)
  assert.equal(calls.length, 0)
  await api.downloadModels(pending, true, signal)
  assert.deepEqual(calls.map(call => call.args.path.model_id), ['lfm3-1.7b-bf16', 'whisper-medium'])
  for (const call of calls) {
    assert.equal(call.name, 'downloadModelV1ModelsModelIdDownloadPost')
    assert.deepEqual(call.args.body, { license_acknowledged: true })
    assert.equal(call.args.signal, signal)
  }
})

test('the fixed Decisions checkpoint can be explicitly downloaded after acknowledgement', async () => {
  const { api, calls, rows } = await settingsApi()
  const models = rows({ laya: checkpoint('laya') })
  await assert.rejects(api.downloadModels(models, false, new AbortController().signal), /acknowledge/)
  assert.equal(calls.length, 0)
  await api.downloadModels(models, true, new AbortController().signal)
  assert.deepEqual(calls.map(call => call.args.path.model_id), ['laya'])
  assert.equal(calls[0].args.body.license_acknowledged, true)
})

test('API details and missing responses become actionable errors', async () => {
  const rejected = await settingsApi({ ok: false, status: 409 }, null, { detail: 'Already downloading' })
  await assert.rejects(rejected.api.fetchModels(new AbortController().signal), /Already downloading/)
  const failed = await settingsApi({ ok: false, status: 503 }, null)
  await assert.rejects(failed.api.downloadModels(failed.rows({ 'whisper-medium': checkpoint('whisper-medium') }), true, new AbortController().signal), /HTTP 503/)
  const offline = await settingsApi(null, null)
  await assert.rejects(offline.api.fetchModels(new AbortController().signal), /Cannot reach the settings API/)
})
