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

function tier(models) {
  return {
    id: 'go', name: 'Go', vram_gb: 8, ram_gb: 32,
    models: Object.entries(models).map(([modality, model]) => ({ modality, model })),
  }
}

async function settingsApi(response = { ok: true }, data = { tier: 'go', tiers: [] }, error = undefined) {
  const calls = []
  const sdk = new SyntheticModule(['getSettings', 'updateSettings'], function () {
    for (const name of ['getSettings', 'updateSettings'])
      this.setExport(name, async args => { calls.push({ name, args }); return { response, data, error } })
  })
  const source = await readFile(new URL('../src/api/settings.ts', import.meta.url), 'utf8')
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link(specifier => { assert.equal(specifier, './generated/sdk.gen'); return sdk })
  await module.evaluate()
  return { api: module.namespace, calls }
}

test('download states distinguish queued, active, finished and failed checkpoints', async () => {
  const { api } = await settingsApi()
  assert.equal(api.downloadState(null), 'unassigned')
  assert.equal(api.downloadState(checkpoint('a')), 'not_downloaded')
  assert.equal(api.downloadState(checkpoint('a', { status: 'downloading' })), 'queued')
  assert.equal(api.downloadState(checkpoint('a', { status: 'downloading', total_bytes: 10 })), 'downloading')
  assert.equal(api.downloadState(checkpoint('a', { status: 'complete' })), 'complete')
  for (const status of ['failed', 'cancelled'])
    assert.equal(api.downloadState(checkpoint('a', { status })), 'failed')
})

test('tier progress weights shared checkpoints once by size', async () => {
  const { api } = await settingsApi()
  const formatter = checkpoint('formatter', { status: 'complete', estimated_bytes: 100 })
  const language = checkpoint('language', { status: 'downloading', downloaded_bytes: 150, total_bytes: 300 })
  const shared = tier({ llm: language, image: null, transcription: formatter, formatting: formatter })
  assert.equal(api.tierProgress(shared), 62)
  assert.equal(api.tierDownloading(shared), true)
  assert.equal(api.tierComplete(shared), false)
  const done = tier({ llm: { ...language, status: 'complete' }, formatting: formatter })
  assert.equal(api.tierProgress(done), 100)
  assert.equal(api.tierComplete(done), true)
  assert.equal(api.tierProgress(tier({ llm: null })), 0)
})

test('license notices are shown only for checkpoints still to download', async () => {
  const { api } = await settingsApi()
  const notice = { license_notice: 'Qwen Community License', license_url: 'https://example.invalid' }
  const notices = api.licenseNotices(tier({
    llm: checkpoint('pending', notice), formatting: checkpoint('done', { ...notice, status: 'complete' }),
  }))
  assert.deepEqual(notices.map(model => model.id), ['pending'])
})

test('choosing a tier acknowledges the shown notices and returns refreshed settings', async () => {
  const { api, calls } = await settingsApi()
  const result = await api.chooseTier('pro', new AbortController().signal)
  assert.equal(result.tier, 'go')
  assert.equal(calls[0].name, 'updateSettings')
  assert.deepEqual(calls[0].args.body, { tier: 'pro', license_acknowledged: true })
})

test('API details and missing responses become actionable errors', async () => {
  const rejected = await settingsApi({ ok: false, status: 422 }, null, { detail: 'Unknown hardware tier: max' })
  await assert.rejects(rejected.api.fetchSettings(new AbortController().signal), /Unknown hardware tier/)
  const failed = await settingsApi({ ok: false, status: 503 }, null)
  await assert.rejects(failed.api.chooseTier('go', new AbortController().signal), /HTTP 503/)
  const offline = await settingsApi(null, null)
  await assert.rejects(offline.api.fetchSettings(new AbortController().signal), /Cannot reach the settings API/)
})


test('incomplete idle tier downloads can resume after a worker restart', async () => {
  const { api, calls } = await settingsApi()
  const selected = tier({
    llm: checkpoint('language'),
    formatting: checkpoint('formatter', { status: 'complete' }),
    image: null,
  })
  assert.equal(api.tierCanDownload(selected), true)
  await api.chooseTier(selected.id, new AbortController().signal)
  assert.deepEqual(calls[0].args.body, { tier: selected.id, license_acknowledged: true })
})

test('download recovery is available for failed or cancelled tiers but not active or complete ones', async () => {
  const { api } = await settingsApi()
  for (const status of ['failed', 'cancelled'])
    assert.equal(api.tierCanDownload(tier({ llm: checkpoint('language', { status }) })), true)
  for (const status of ['downloading', 'cancelling', 'complete'])
    assert.equal(api.tierCanDownload(tier({ llm: checkpoint('language', { status }) })), false)
  assert.equal(api.tierCanDownload(tier({
    llm: checkpoint('language'),
    formatting: checkpoint('formatter', { status: 'downloading' }),
  })), false)
})
