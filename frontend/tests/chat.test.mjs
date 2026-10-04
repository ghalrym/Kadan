// No additional test dependencies: compile the SDK adapter into an isolated temp
// directory, then exercise its real HTTP request path with controlled fetch.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { after, test } from 'node:test'

const output = mkdtempSync(join(tmpdir(), 'kadan-chat-tests-'))
after(() => rmSync(output, { recursive: true, force: true }))
execFileSync(
  process.execPath,
  [
    resolve('node_modules/typescript/bin/tsc'),
    '--ignoreConfig',
    'src/api/chat.ts',
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
const {
  requestChat,
  chatRequest,
  MAX_MESSAGE_LENGTH,
  MAX_HISTORY_MESSAGES,
} = require(join(output, 'chat.js'))
const { client } = require(join(output, 'generated/client.gen.js'))
const messages = [{ role: 'user', text: 'Hello', meta: 'not sent' }]
const signal = () => new AbortController().signal

function mockFetch(handler) {
  client.setConfig({ baseUrl: 'http://kadan.test', fetch: handler })
}

test('SDK sends real custom contract and history, without forcing a model', async () => {
  mockFetch(async (request) => {
    assert.equal(request.url, 'http://kadan.test/v1/chat/completions')
    assert.equal(request.method, 'POST')
    assert.equal(request.headers.get('Content-Type'), 'application/json')
    assert.deepEqual(await request.json(), {
      messages: [{ role: 'user', text: 'Hello' }],
    })
    return Response.json({
      message: { role: 'assistant', text: 'World', meta: 'runtime' },
    })
  })
  assert.deepEqual(await requestChat(messages, signal()), {
    role: 'assistant',
    text: 'World',
    meta: 'runtime',
  })
  assert.deepEqual(
    chatRequest([
      ...messages,
      { role: 'assistant', text: 'World' },
      { role: 'user', text: 'Again' },
    ]).messages.map((m) => m.text),
    ['Hello', 'World', 'Again'],
  )
})

test('unavailable, busy, validation and server errors are not successes; retry can succeed', async () => {
  for (const status of [503, 409, 429, 422, 500]) {
    mockFetch(async () => Response.json({ detail: 'failure' }, { status }))
    await assert.rejects(requestChat(messages, signal()))
  }
  mockFetch(async () =>
    Response.json({ message: { role: 'assistant', text: 'Recovered' } }),
  )
  assert.equal((await requestChat(messages, signal())).text, 'Recovered')
})

test('network failure and malformed successes are rejected', async () => {
  mockFetch(async () => {
    throw new TypeError('Network offline')
  })
  await assert.rejects(requestChat(messages, signal()))
  for (const body of [
    {},
    { message: { role: 'user', text: 'wrong role' } },
    { message: { role: 'assistant', text: '' } },
  ]) {
    mockFetch(async () => Response.json(body))
    await assert.rejects(
      requestChat(messages, signal()),
      /invalid chat response/,
    )
  }
})

test('cancellation reaches fetch and prevents a successful completion', async () => {
  const controller = new AbortController()
  mockFetch(
    (request) =>
      new Promise((_, reject) => {
        request.signal.addEventListener(
          'abort',
          () => reject(request.signal.reason),
          { once: true },
        )
      }),
  )
  const pending = requestChat(messages, controller.signal)
  await new Promise((resolve) => setImmediate(resolve))
  controller.abort()
  await assert.rejects(pending)
})

test('bounds reject empty, oversized input and oversized history before fetch', () => {
  assert.throws(() => chatRequest([]))
  assert.throws(() => chatRequest([{ role: 'user', text: ' ' }]))
  assert.throws(() =>
    chatRequest([{ role: 'user', text: 'x'.repeat(MAX_MESSAGE_LENGTH + 1) }]),
  )
  assert.throws(() =>
    chatRequest(Array(MAX_HISTORY_MESSAGES + 1).fill(messages[0])),
  )
  assert.equal(
    chatRequest(Array(MAX_HISTORY_MESSAGES).fill(messages[0])).messages.length,
    MAX_HISTORY_MESSAGES,
  )
})
