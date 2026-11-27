import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'

async function transcriptionApi(result) {
  const calls = []
  const sdk = new SyntheticModule(['transcribeAudio'], function () {
    this.setExport('transcribeAudio', async args => { calls.push({ name: 'transcribeAudio', ...args }); return result })
  })
  const source = await readFile(new URL('../api/transcription.ts', import.meta.url), 'utf8')
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link(specifier => { assert.equal(specifier, './generated/sdk.gen'); return sdk })
  await module.evaluate()
  return { api: module.namespace, calls }
}

const signal = new AbortController().signal

test('transcription submits trimmed inline audio and returns the transcript', async () => {
  const { api, calls } = await transcriptionApi({ response: { ok: true }, data: { text: 'Pay $1.25.', language: 'en' } })
  assert.deepEqual(await api.requestTranscription(' data:audio/wav;base64,AAAA ', signal), { text: 'Pay $1.25.', language: 'en' })
  assert.deepEqual(calls[0].body, { audio: 'data:audio/wav;base64,AAAA' })
  assert.equal(calls[0].signal, signal)
})

test('silent transcripts stay empty', async () => {
  const { api } = await transcriptionApi({ response: { ok: true }, data: { text: '' } })
  assert.equal((await api.requestTranscription('data:audio/wav;base64,AAAA', signal)).text, '')
})

test('failed transcription preserves backend error details', async () => {
  const { api } = await transcriptionApi({ response: { ok: false }, error: { detail: 'Speech worker offline' } })
  await assert.rejects(api.requestTranscription('data:audio/wav;base64,AAAA', signal), /Speech worker offline/)
})
