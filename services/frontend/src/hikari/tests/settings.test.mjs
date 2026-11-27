import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'

function checkpoint(id, overrides = {}) {
  return {
    id, kind: id.startsWith('inkling-2-mini') ? 'llm' : 'other',
    display_name: id, status: 'not_downloaded', downloaded_bytes: 0, total_bytes: 0,
    estimated_bytes: 100, license_notice: null, license_url: null, error: null, ...overrides,
  }
}

async function load(path) {
  return new SourceTextModule(stripTypeScriptTypes(await readFile(new URL(path, import.meta.url), 'utf8')))
}

async function settingsApi(response = { ok: true }, data = { models: [] }, error = undefined) {
  const calls = []
  const names = ['listModelsV1ModelsGet', 'downloadModelV1ModelsModelIdDownloadPost', 'updateSettings']
  const sdk = new SyntheticModule(names, function () {
    for (const name of names)
      this.setExport(name, async args => { calls.push({ name, args }); return { response, data, error } })
  })
  const models = await load('../data/models.ts')
  await models.link(() => assert.fail('models.ts has no imports'))
  const module = await load('../api/settings.ts')
  await module.link(specifier => ({ './generated/sdk.gen': sdk, '../data/models': models })[specifier])
  await module.evaluate()
  const capabilities = models.namespace.capabilities
  const rows = overrides => capabilities.map(capability => ({
    capability, model: capability.id in overrides ? overrides[capability.id] : checkpoint(capability.id === 'inkling-2-mini' ? 'inkling-2-mini-iq4-nl' : capability.id, { status: 'complete' }),
  }))
  return { api: module.namespace, calls, capabilities, rows }
}

test('every capability reads its checkpoint from the model snapshot including active Inkling quant', async () => {
  const { api, calls, capabilities } = await settingsApi({ ok: true }, {
    selected_model_id: 'inkling-2-mini-q8-0',
    models: [
      checkpoint('inkling-2-mini-q8-0'),
      checkpoint('laya'),
      checkpoint('flux.2-klein-4b'),
      checkpoint('qwen-tts-1.7b-custom'),
    ],
  })
  const result = await api.fetchModels(new AbortController().signal)
  assert.deepEqual(result.map(row => row.capability.label),
    ['Language & Decisions', 'Image', 'Speech to text', 'Text to speech'])
  assert.equal(result[0].model.id, 'inkling-2-mini-q8-0')
  assert.equal(result[1].model.id, 'flux.2-klein-4b')
  assert.equal(result[3].model.id, 'qwen-tts-1.7b-custom')
  assert.equal(result[2].model, null)
  assert.equal(capabilities.length, 4)
  assert.equal(result.some(row => row.model?.id === 'laya'), false)
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
  const language = checkpoint('inkling-2-mini-iq4-nl', { status: 'downloading', downloaded_bytes: 150, total_bytes: 300 })
  const active = rows({ 'inkling-2-mini': language, 'flux.2-klein-4b': null })
  assert.equal(api.modelsProgress(active), 70)
  assert.equal(api.modelsDownloading(active), true)
  assert.equal(api.modelsComplete(active), false)
  assert.equal(api.modelsComplete(rows({})), true)
  assert.equal(api.modelsComplete(rows({ 'qwen3-asr-1.7b': null })), false)
})

test('license notices are shown only for checkpoints still to download', async () => {
  const { api, rows } = await settingsApi()
  const notice = { license_notice: 'Apache 2.0 License', license_url: 'https://example.invalid' }
  const notices = api.licenseNotices(rows({
    'inkling-2-mini': checkpoint('inkling-2-mini-iq4-nl', notice),
    'qwen3-asr-1.7b': checkpoint('qwen3-asr-1.7b', { ...notice, status: 'complete' }),
  }))
  assert.deepEqual(notices.map(model => model.id), ['inkling-2-mini-iq4-nl'])
})

test('downloads are offered for missing, failed or cancelled checkpoints but not active ones', async () => {
  const { api, rows } = await settingsApi()
  for (const status of ['not_downloaded', 'failed', 'cancelled'])
    assert.equal(api.modelsCanDownload(rows({ 'qwen3-asr-1.7b': checkpoint('qwen3-asr-1.7b', { status }) })), true)
  for (const status of ['downloading', 'cancelling', 'complete'])
    assert.equal(api.modelsCanDownload(rows({ 'qwen3-asr-1.7b': checkpoint('qwen3-asr-1.7b', { status }) })), false)
})

test('downloading requires acknowledgement and queues only idle incomplete checkpoints', async () => {
  const { api, calls, rows } = await settingsApi({ ok: true }, { models: [] })
  const signal = new AbortController().signal
  const pending = rows({
    'inkling-2-mini': checkpoint('inkling-2-mini-iq4-nl'),
    'flux.2-klein-4b': checkpoint('flux.2-klein-4b', { status: 'downloading' }),
    'qwen3-asr-1.7b': checkpoint('qwen3-asr-1.7b', { status: 'failed' }),
  })
  await assert.rejects(api.downloadModels(pending, false, signal), /acknowledge the model licenses/)
  assert.equal(calls.length, 0)
  await api.downloadModels(pending, true, signal)
  assert.deepEqual(calls.map(call => call.args.path.model_id), ['inkling-2-mini-iq4-nl', 'qwen3-asr-1.7b'])
  for (const call of calls) {
    assert.equal(call.name, 'downloadModelV1ModelsModelIdDownloadPost')
    assert.deepEqual(call.args.body, { license_acknowledged: true })
    assert.equal(call.args.signal, signal)
  }
})

test('the shared Inkling language checkpoint can be explicitly downloaded after acknowledgement', async () => {
  const { api, calls, rows } = await settingsApi()
  const models = rows({ 'inkling-2-mini': checkpoint('inkling-2-mini-iq4-nl') })
  await assert.rejects(api.downloadModels(models, false, new AbortController().signal), /acknowledge/)
  assert.equal(calls.length, 0)
  await api.downloadModels(models, true, new AbortController().signal)
  assert.deepEqual(calls.map(call => call.args.path.model_id), ['inkling-2-mini-iq4-nl'])
  assert.equal(calls[0].args.body.license_acknowledged, true)
})

test('API details and missing responses become actionable errors', async () => {
  const rejected = await settingsApi({ ok: false, status: 409 }, null, { detail: 'Already downloading' })
  await assert.rejects(rejected.api.fetchModels(new AbortController().signal), /Already downloading/)
  const failed = await settingsApi({ ok: false, status: 503 }, null)
  await assert.rejects(failed.api.downloadModels(failed.rows({ 'qwen3-asr-1.7b': checkpoint('qwen3-asr-1.7b') }), true, new AbortController().signal), /HTTP 503/)
  const offline = await settingsApi(null, null)
  await assert.rejects(offline.api.fetchModels(new AbortController().signal), /Cannot reach the settings API/)
})

test('speech to text uses fixed Qwen3-ASR 1.7B checkpoint and requires acknowledged download', async () => {
  const stt = checkpoint('qwen3-asr-1.7b', { license_notice: 'Apache 2.0 License' })
  const { api, calls, capabilities } = await settingsApi({ ok: true }, { models: [stt] })
  const rows = api.capabilityModels([stt])
  assert.equal(rows.length, capabilities.length)
  assert.equal(rows[0].capability.id, 'inkling-2-mini')
  assert.equal(rows.find(row => row.capability.id === stt.id).capability.label, 'Speech to text')
  assert.deepEqual(api.licenseNotices(rows).map(model => model.id), [stt.id])
  const signal = new AbortController().signal
  await assert.rejects(api.downloadModels(rows, false, signal), /acknowledge/)
  assert.equal(calls.length, 0)
  await api.downloadModels(rows, true, signal)
  assert.deepEqual(calls.map(call => call.args.path.model_id), [stt.id])
  assert.equal(calls[0].args.body.license_acknowledged, true)
  assert.equal(api.capabilityModels([]).find(row => row.capability.id === stt.id).model, null)
})

test('language model variant helpers and updateLlmModel switch between Q4/Q8 and Standard/Abliterated', async () => {
  const { api, calls } = await settingsApi({ ok: true }, {
    defaults: { llm: 'inkling-2-mini-abliterated-q8-0' },
    models: [checkpoint('inkling-2-mini-abliterated-q8-0')],
  })
  assert.deepEqual(api.llmVariantParts('inkling-2-mini-iq4-nl'), { quant: 'q4', abliterated: false })
  assert.deepEqual(api.llmVariantParts('inkling-2-mini-q8-0'), { quant: 'q8', abliterated: false })
  assert.deepEqual(api.llmVariantParts('inkling-2-mini-abliterated-iq4-nl'), { quant: 'q4', abliterated: true })
  assert.deepEqual(api.llmVariantParts('inkling-2-mini-abliterated-q8-0'), { quant: 'q8', abliterated: true })
  assert.equal(api.llmModelId('q4', false), 'inkling-2-mini-iq4-nl')
  assert.equal(api.llmModelId('q8', false), 'inkling-2-mini-q8-0')
  assert.equal(api.llmModelId('q4', true), 'inkling-2-mini-abliterated-iq4-nl')
  assert.equal(api.llmModelId('q8', true), 'inkling-2-mini-abliterated-q8-0')

  const signal = new AbortController().signal
  const next = await api.updateLlmModel('inkling-2-mini-abliterated-q8-0', signal)
  assert.equal(calls[0].name, 'updateSettings')
  assert.deepEqual(calls[0].args.body, { llm_model: 'inkling-2-mini-abliterated-q8-0' })
  assert.equal(next[0].model.id, 'inkling-2-mini-abliterated-q8-0')
})
