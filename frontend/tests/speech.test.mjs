// No additional test dependencies: compile the SDK adapter into an isolated temp
// directory, then exercise its real HTTP request path with controlled fetch.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { after, test } from 'node:test'

const output = mkdtempSync(join(tmpdir(), 'kadan-speech-tests-'))
after(() => rmSync(output, { recursive: true, force: true }))
execFileSync(
  process.execPath,
  [
    resolve('node_modules/typescript/bin/tsc'),
    '--ignoreConfig',
    'src/api/speech.ts',
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
const { speechRequest, requestSpeech, fetchSpeechHistory } = require(
  join(output, 'speech.js'),
)
const { client } = require(join(output, 'generated/client.gen.js'))
const signal = () => new AbortController().signal
function mockFetch(fetch) {
  client.setConfig({ baseUrl: 'http://kadan.test', fetch })
}

test('describe and clone use discriminated payloads through generated SDK', async () => {
  for (const mode of ['describe', 'clone']) {
    const body = speechRequest(' Hello ', mode, ' Voice ')
    assert.deepEqual(body, {
      script: 'Hello',
      voice:
        mode === 'clone'
          ? { mode, sample: 'Voice' }
          : { mode, description: 'Voice' },
    })
    mockFetch(async (request) => {
      assert.equal(request.url, 'http://kadan.test/v1/audio/speech')
      assert.equal(request.method, 'POST')
      assert.deepEqual(await request.json(), body)
      return Response.json({ detail: 'No provider' }, { status: 503 })
    })
    await assert.rejects(requestSpeech(body, signal()), /No speech provider/)
  }
})

test('history fetch is empty and failures are explicit', async () => {
  mockFetch(async (request) => {
    assert.equal(request.method, 'GET')
    return Response.json({ audio: [], script: '', voice_description: '' })
  })
  assert.deepEqual(await fetchSpeechHistory(signal()), [])
  mockFetch(async () => Response.json({ detail: 'Failure' }, { status: 500 }))
  await assert.rejects(fetchSpeechHistory(signal()), /HTTP 500/)
  mockFetch(async () => Response.json({ audio: [{}] }))
  await assert.rejects(fetchSpeechHistory(signal()), /invalid history/)
})

test('validation, network and malformed success are never fake successes', async () => {
  assert.throws(() => speechRequest(' ', 'describe', 'Voice'))
  assert.throws(() => speechRequest('Hello', 'clone', ' '))
  const body = speechRequest('Hello', 'describe', 'Warm')
  for (const status of [422, 500]) {
    mockFetch(async () => Response.json({}, { status }))
    await assert.rejects(requestSpeech(body, signal()))
  }
  mockFetch(async () => {
    throw new TypeError('offline')
  })
  await assert.rejects(requestSpeech(body, signal()), /Cannot reach/)
  mockFetch(async () => Response.json({ audio: {} }))
  await assert.rejects(requestSpeech(body, signal()), /invalid response/)
})

test('abort propagates through speech SDK', async () => {
  const controller = new AbortController()
  mockFetch(
    (request) =>
      new Promise((_, reject) =>
        request.signal.addEventListener(
          'abort',
          () => reject(request.signal.reason),
          { once: true },
        ),
      ),
  )
  const pending = requestSpeech(
    speechRequest('Hello', 'describe', 'Warm'),
    controller.signal,
  )
  await new Promise((resolve) => setImmediate(resolve))
  controller.abort()
  await assert.rejects(pending)
})

test('long script and voice are sent unchanged without arbitrary caps', async () => {
  for (const mode of ['describe', 'clone']) {
    const body = speechRequest('x'.repeat(20000), mode, 'v'.repeat(10000))
    mockFetch(async request => {
      assert.deepEqual(await request.json(), body)
      return Response.json({ detail: 'No provider' }, { status: 503 })
    })
    await assert.rejects(requestSpeech(body, signal()), /No speech provider/)
  }
})
