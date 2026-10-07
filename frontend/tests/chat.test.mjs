// No additional test dependencies: compile the SDK adapter into an isolated temp
// directory, then exercise its real HTTP request path with controlled fetch.
// These contract tests do not validate browser layout or GPU/model inference.
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
const { requestChat, chatRequest } = require(join(output, 'chat.js'))
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
  for (const status of [503, 409, 429, 413, 422, 500]) {
    mockFetch(async () => Response.json({ detail: 'failure' }, { status }))
    await assert.rejects(
      requestChat(messages, signal()),
      status === 413 ? /failure/ : undefined,
    )
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

test('empty conversations and blank messages are rejected', () => {
  assert.throws(() => chatRequest([]))
  assert.throws(() => chatRequest([{ role: 'user', text: ' \n\t' }]))
})

test('large skill text and long histories are forwarded intact without product caps', async () => {
  const skill =
    '  # Skill\n' + 'Detailed instructions.\n'.repeat(50_000) + '\n  '
  const history = Array.from({ length: 64 }, (_, index) => ({
    role: index % 2 ? 'assistant' : 'user',
    text: `Turn ${index}`,
  }))
  history.push({ role: 'user', text: skill })
  mockFetch(async (request) => {
    assert.deepEqual(await request.json(), { messages: history })
    return Response.json({ message: { role: 'assistant', text: 'Received' } })
  })
  assert.equal((await requestChat(history, signal())).text, 'Received')
})

test('model context detail is surfaced and missing detail gives actionable context guidance', async () => {
  const detail =
    'Prompt uses 17000 tokens; loaded model context is 16384 tokens.'
  mockFetch(async () => Response.json({ detail }, { status: 413 }))
  await assert.rejects(requestChat(messages, signal()), { message: detail })
  mockFetch(async () => Response.json({}, { status: 413 }))
  await assert.rejects(
    requestChat(messages, signal()),
    /configured token context/,
  )
})

const { streamChat } = require(join(output, 'chat.js'))
const originalFetch = globalThis.fetch
after(() => {
  globalThis.fetch = originalFetch
})
const frame = (delta, finish_reason = null) =>
  'data: ' +
  JSON.stringify({
    object: 'chat.completion.chunk',
    choices: [{ index: 0, delta, finish_reason }],
  }) +
  '\n\n'

test('SSE renders before completion, survives byte-fragmented Unicode and CRLF', async () => {
  let controller
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/v1/chat/completions')
    assert.equal(JSON.parse(options.body).stream, true)
    return new Response(
      new ReadableStream({
        start(c) {
          controller = c
        },
      }),
      { headers: { 'Content-Type': 'text/event-stream' } },
    )
  }
  const updates = []
  let finished = false
  const request = streamChat(messages, signal(), (text) =>
    updates.push(text),
  ).then((x) => {
    finished = true
    return x
  })
  await new Promise((resolve) => setImmediate(resolve))
  const bytes = new TextEncoder().encode(
    frame({ content: '你好 ' }).replaceAll('\n', '\r\n'),
  )
  for (const byte of bytes) controller.enqueue(new Uint8Array([byte]))
  await new Promise((resolve) => setImmediate(resolve))
  assert.deepEqual(updates, ['你好 '])
  assert.equal(finished, false)
  controller.enqueue(
    new TextEncoder().encode(
      frame({ content: 'world' }) + frame({}, 'stop') + 'data: [DONE]\n\n',
    ),
  )
  controller.close()
  assert.equal((await request).text, '你好 world')
})

test('SSE errors, premature EOF and missing terminal chunk cannot become success', async () => {
  for (const data of [
    frame({ content: 'partial' }),
    frame({ content: 'partial' }) + 'data: [DONE]\n\n',
    'data: {"error":{"message":"native failed"}}\n\n' + 'data: [DONE]\n\n',
  ]) {
    globalThis.fetch = async () =>
      new Response(data, { headers: { 'Content-Type': 'text/event-stream' } })
    await assert.rejects(streamChat(messages, signal(), () => {}))
  }
})

test('SSE accepts length terminal and rejects malformed chunks', async () => {
  globalThis.fetch = async () =>
    new Response(
      frame({ content: 'answer' }) + frame({}, 'length') + 'data: [DONE]\n\n',
      { headers: { 'Content-Type': 'text/event-stream' } },
    )
  assert.equal((await streamChat(messages, signal(), () => {})).text, 'answer')
  globalThis.fetch = async () =>
    new Response('data: {}\n\n', {
      headers: { 'Content-Type': 'text/event-stream' },
    })
  await assert.rejects(
    streamChat(messages, signal(), () => {}),
    /invalid chat stream/,
  )
})


test('JSON and SSE preserve terminal cache diagnostics without sending them as history', async () => {
  const cache = { hit: true, reused_tokens: 24, stored_tokens: 48, host_bytes: 65536,
    device_bytes: { '0': 4096 }, reason: 'hit', retention_reason: 'retained', limit_bytes: 2147483648 }
  mockFetch(async () => Response.json({ message: { role: 'assistant', text: 'Four.' }, cache }))
  const json = await requestChat(messages, signal())
  assert.deepEqual(json.cache, cache)
  const terminal = JSON.parse(frame({}, 'stop').slice(6).trim())
  terminal.cache = cache
  globalThis.fetch = async () => new Response(
    frame({ content: 'Four.' }) + 'data: ' + JSON.stringify(terminal) + '\n\n' + 'data: [DONE]\n\n',
    { headers: { 'Content-Type': 'text/event-stream' } },
  )
  const streamed = await streamChat(messages, signal(), () => {}, 'conversation')
  assert.deepEqual(streamed.cache, cache)
  assert.deepEqual(chatRequest([streamed]), { messages: [{ role: 'assistant', text: 'Four.' }] })
})
