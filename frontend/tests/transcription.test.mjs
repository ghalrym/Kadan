// No additional test dependencies: compile the SDK adapter into an isolated temp
// directory, then exercise its real HTTP request path with controlled fetch.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { after, test } from 'node:test'

const output = mkdtempSync(join(tmpdir(), 'kadan-transcription-tests-'))
after(() => rmSync(output, { recursive: true, force: true }))
execFileSync(
  process.execPath,
  [
    resolve('node_modules/typescript/bin/tsc'),
    '--ignoreConfig',
    'src/api/transcription.ts',
    '--outDir',
    output,
    '--module',
    'commonjs',
    '--target',
    'es2023',
    '--lib',
    'es2023,dom',
    '--skipLibCheck',
    '--esModuleInterop',
  ],
  { stdio: 'inherit' },
)
writeFileSync(join(output, 'package.json'), '{"type":"commonjs"}')
const require = createRequire(import.meta.url)
const { requestTranscription } = require(join(output, 'transcription.js'))
const { client } = require(join(output, 'generated/client.gen.js'))
function mockFetch(fetch) { client.setConfig({ baseUrl: 'http://kadan.test', fetch }) }
const signal = () => new AbortController().signal

test('typed endpoint sends reference and formatting without uploading files', async () => {
  mockFetch(async request => {
    assert.equal(request.url, 'http://kadan.test/v1/audio/transcriptions')
    assert.equal(request.method, 'POST')
    assert.deepEqual(await request.json(), { audio: 'reference', formatting: false })
    return Response.json({ text: 'Controlled future-provider response' })
  })
  assert.equal(await requestTranscription(' reference ', false, signal()), 'Controlled future-provider response')
})
test('unavailable, validation, network, and malformed responses never produce fixture text', async () => {
  mockFetch(async () => Response.json({ detail: 'Unavailable' }, { status: 503 }))
  await assert.rejects(requestTranscription('reference', true, signal()), /no speech-to-text provider/)
  for (const status of [422, 500]) {
    mockFetch(async () => Response.json({}, { status }))
    await assert.rejects(requestTranscription('reference', true, signal()))
  }
  mockFetch(async () => { throw new Error('offline') })
  await assert.rejects(requestTranscription('reference', true, signal()))
  for (const value of [{}, { text: '' }, { text: 42 }]) {
    mockFetch(async () => Response.json(value))
    await assert.rejects(requestTranscription('reference', true, signal()), /invalid transcript/)
  }
})
test('blank or overlong reference fails before fetch', async () => {
  mockFetch(async () => { assert.fail('must not fetch') })
  for (const value of [' ', 'x'.repeat(2049)]) await assert.rejects(requestTranscription(value, true, signal()), /1–2048/)
})
test('cancel reaches SDK fetch', async () => {
  const controller = new AbortController()
  mockFetch(request => new Promise((_, reject) => {
    request.signal.addEventListener('abort', () => reject(request.signal.reason), { once: true })
  }))
  const pending = requestTranscription('reference', true, controller.signal)
  await new Promise(resolve => setImmediate(resolve))
  controller.abort()
  await assert.rejects(pending)
})
