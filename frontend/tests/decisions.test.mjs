// No additional test dependencies: compile the decisions adapter into an isolated
// temp directory, then exercise its real HTTP request path with controlled fetch.
// These contract tests do not validate browser layout or CPU/model inference.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { after, test } from 'node:test'

const output = mkdtempSync(join(tmpdir(), 'kadan-decisions-tests-'))
after(() => rmSync(output, { recursive: true, force: true }))
execFileSync(
  process.execPath,
  [
    resolve('node_modules/typescript/bin/tsc'),
    '--ignoreConfig',
    'src/api/decisions.ts',
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
const { requestDecisions, validateDecisionRequest, DecisionError } = require(join(output, 'decisions.js'))
const { client } = require(join(output, 'generated/client.gen.js'))
const signal = () => new AbortController().signal
const questions = [
  { key: 'intent', instructions: 'What is wanted?', type: 'Choice', options: [{ key: 'refund', description: '' }, { key: 'exchange', description: '' }] },
  { key: 'escalate', instructions: 'Needs a manager', type: 'Noul', trueWhen: '', falseWhen: '' },
]
const answers = [
  { key: 'intent', type: 'Choice', value: 'refund', confidence: 0.8, probabilities: { refund: 0.8, exchange: 0.2 } },
  { key: 'escalate', type: 'Noul', value: 0.3 },
]

function mockFetch(handler) {
  client.setConfig({ baseUrl: 'http://kadan.test', fetch: handler })
}
async function rejection(promise) {
  try {
    await promise
  } catch (error) {
    return error
  }
  assert.fail('expected a rejection')
}

test('sends the typed request and returns answers in question order', async () => {
  mockFetch(async (request) => {
    assert.equal(request.url, 'http://kadan.test/v1/decisions')
    assert.equal(request.method, 'POST')
    assert.deepEqual(await request.json(), { state: 'Broken', questions })
    return Response.json({ answers })
  })
  assert.deepEqual(await requestDecisions('Broken', questions, signal()), answers)
})

test('local validation scopes problems to questions and never calls the API', async () => {
  mockFetch(async () => assert.fail('must not send'))
  const issues = validateDecisionRequest(' ', [
    { key: 'a', instructions: '', type: 'Choice', options: [{ key: 'x', description: '' }, { key: 'x', description: '' }] },
    { key: 'a', instructions: 'ok', type: 'Score', levels: ['low', ' '] },
  ])
  assert.deepEqual(issues, [
    { message: 'Describe the state to evaluate.' },
    { message: 'Instructions are required.', question: 0 },
    { message: 'Option key “x” is used more than once.', question: 0 },
    { message: 'Key “a” is already used by another question.', question: 1 },
    { message: 'Every rubric level needs a description.', question: 1 },
  ])
  assert.deepEqual(validateDecisionRequest('ok', []), [{ message: 'Add at least one question.' }])
  const error = await rejection(requestDecisions('', questions, signal()))
  assert(error instanceof DecisionError)
  assert.equal(error.retryable, false)
})

test('API validation errors become readable, question-scoped issues', async () => {
  mockFetch(async () =>
    Response.json({
      detail: [
        { loc: ['body', 'questions', 0, 'Choice', 'options', 1, 'key'], msg: 'String should have at least 1 character', type: 'x' },
        { loc: ['body'], msg: 'Value error, Instructions for question "escalate" must not be blank', type: 'x' },
        { loc: ['body', 'state'], msg: 'Too long', type: 'x' },
      ],
    }, { status: 422 }),
  )
  const error = await rejection(requestDecisions('Broken', questions, signal()))
  assert.equal(error.title, 'Check your inputs')
  assert.deepEqual(error.issues, [
    { question: 0, message: 'Option 2 key: String should have at least 1 character' },
    { question: 1, message: 'Instructions for question "escalate" must not be blank' },
    { message: 'State: Too long' },
  ])
})

test('runtime, proxy, network and malformed failures are titled and retryable as appropriate', async () => {
  const cases = [
    [() => Response.json({ detail: 'Question too long for the token budget' }, { status: 422 }), 'Input too long', false, /token budget/],
    [() => Response.json({ detail: 'Laya checkpoint is incomplete' }, { status: 503 }), 'Decision model unavailable', true, /checkpoint/],
    [() => new Response('<html>Bad gateway</html>', { status: 502, headers: { 'Content-Type': 'text/html' } }), 'Unexpected model output', true, /could not read/],
    [() => Response.json({}, { status: 500 }), 'Evaluation failed', true, /HTTP 500/],
    [() => { throw new TypeError('fetch failed') }, 'Cannot reach Kadan', true, /backend is running/],
    [() => Response.json({ answers: [answers[1], answers[0]] }), 'Unexpected model output', true, /do not match/],
  ]
  for (const [respond, title, retryable, message] of cases) {
    mockFetch(async () => respond())
    const error = await rejection(requestDecisions('Broken', questions, signal()))
    assert(error instanceof DecisionError, title)
    assert.equal(error.title, title)
    assert.equal(error.retryable, retryable, title)
    assert.match(error.message, message)
  }
})

test('an aborted request rejects without a DecisionError', async () => {
  const controller = new AbortController()
  mockFetch(async (request) => {
    controller.abort()
    throw request.signal.reason ?? new DOMException('Aborted', 'AbortError')
  })
  const error = await rejection(requestDecisions('Broken', questions, controller.signal))
  assert(!(error instanceof DecisionError))
})
