import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'

async function transcriptionApi(result) {
  const calls = []
  const sdk = new SyntheticModule(['getSettings', 'updateSettings', 'transcribeAudio'], function () {
    for (const name of ['getSettings', 'updateSettings', 'transcribeAudio'])
      this.setExport(name, async args => { calls.push({ name, ...args }); return result })
  })
  const source = await readFile(new URL('../src/api/transcription.ts', import.meta.url), 'utf8')
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link(specifier => { assert.equal(specifier, './generated/sdk.gen'); return sdk })
  await module.evaluate()
  return { api: module.namespace, calls }
}

const signal = new AbortController().signal

test('transcription uses saved preference and displays the returned text', async () => {
  const { api, calls } = await transcriptionApi({ response: { ok: true }, data: { text: 'Pay $1.25.', raw_text: '  Pay  $1.25. ' } })
  assert.deepEqual(await api.requestTranscription(' data:audio/wav;base64,AAAA ', signal), { text: 'Pay $1.25.', raw_text: '  Pay  $1.25. ' })
  assert.deepEqual(calls[0].body, { audio: 'data:audio/wav;base64,AAAA' })
  assert.equal(calls[0].signal, signal)
})

test('silent transcripts stay empty', async () => {
  const { api } = await transcriptionApi({ response: { ok: true }, data: { text: '' } })
  assert.equal((await api.requestTranscription('data:audio/wav;base64,AAAA', signal)).text, '')
})

test('reads and saves both formatting states without changing models', async () => {
  for (const enabled of [true, false]) {
    const { api, calls } = await transcriptionApi({ response: { ok: true }, data: { whisper_formatting: enabled } })
    assert.equal(await api.requestTranscriptionFormatting(signal), enabled)
    assert.equal(calls[0].name, 'getSettings')
    assert.equal(await api.requestTranscriptionFormatting(signal, enabled), enabled)
    assert.deepEqual(calls[1].body, { models: {}, whisper_formatting: enabled })
    assert.equal(calls[1].name, 'updateSettings')
  }
})

test('failed saves keep the backend error', async () => {
  const { api } = await transcriptionApi({ response: { ok: false }, error: { detail: 'Cannot save setting' } })
  await assert.rejects(api.requestTranscriptionFormatting(signal, false), /Cannot save setting/)
})

test('missing settings are unavailable instead of a fabricated enabled state', async () => {
  const { api } = await transcriptionApi({ response: { ok: true }, data: { whisper_formatting: null } })
  await assert.rejects(api.requestTranscriptionFormatting(signal), /unavailable/)
})

test('formatting failure preserves raw text and the warning for display', async () => {
  const response = { text: 'um do not send it', raw_text: 'um do not send it', formatting_status: 'failed', formatting_error: 'Formatter unavailable' }
  const { api } = await transcriptionApi({ response: { ok: true }, data: response })
  assert.deepEqual(await api.requestTranscription('data:audio/wav;base64,AAAA', signal), response)
})
