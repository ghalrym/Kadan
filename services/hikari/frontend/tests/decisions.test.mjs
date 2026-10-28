import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'

const score = (count) => ({ key: 'priority', type: 'Score', instructions: 'Rate urgency.', levels: Array.from({ length: count }, (_, i) => `Level ${i}`) })
const choice = { key: 'intent', type: 'Choice', instructions: 'Pick intent.', options: [{ key: 'help', description: 'Needs help' }] }

async function decisionsApi(result = { response: { ok: true }, data: { answers: [{ key: 'intent', type: 'Choice', value: 'help' }] } }) {
  const calls = []
  const sdk = new SyntheticModule(['evaluateDecisions'], function () {
    this.setExport('evaluateDecisions', async (args) => { calls.push(args); return result })
  })
  const source = await readFile(new URL('../src/api/decisions.ts', import.meta.url), 'utf8')
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link((specifier) => { assert.equal(specifier, './generated/sdk.gen'); return sdk })
  await module.evaluate()
  return { api: module.namespace, calls }
}

test('omitted model preserves the Laya payload and state limit', async () => {
  const { api, calls } = await decisionsApi()
  const signal = new AbortController().signal
  await api.requestDecisions('Customer asks for help.', [choice], signal)
  assert.deepEqual(calls[0].body, { state: 'Customer asks for help.', questions: [choice] })
  assert.equal(calls[0].signal, signal)
  await assert.rejects(api.requestDecisions('x'.repeat(8001), [choice], signal), /Some fields need attention/)
  assert.equal(calls.length, 1)
})

test('explicit D1 sends its model and long state unchanged', async () => {
  const { api, calls } = await decisionsApi()
  const state = 'document '.repeat(3000)
  const signal = new AbortController().signal
  await api.requestDecisions(state, [choice], signal, 'd1-2b-bf16')
  assert.deepEqual(calls[0].body, { state, questions: [choice], model: 'd1-2b-bf16' })
  assert.equal(calls[0].signal, signal)
})

test('D1 validates Score bounds without changing or clamping rubric levels', async () => {
  const { api } = await decisionsApi()
  for (const count of [1, 11]) {
    const question = score(count)
    const issues = api.validateDecisionRequest('state', [question], 'd1-2b-bf16')
    assert.equal(issues[0].question, 0)
    assert.match(issues[0].message, /2–10/)
    assert.equal(question.levels.length, count)
  }
  for (const count of [2, 10])
    assert.deepEqual(api.validateDecisionRequest('state', [score(count)], 'd1-2b-bf16'), [])
  for (const count of [1, 64])
    assert.deepEqual(api.validateDecisionRequest('state', [score(count)], 'laya'), [])
})

test('D1 preserves structural validation and limits Choice to 64 options', async () => {
  const { api, calls } = await decisionsApi()
  const question = { ...choice, options: Array.from({ length: 65 }, (_, i) => ({ key: `option-${i}`, description: '' })) }
  assert.match(api.validateDecisionRequest('state', [question], 'd1-2b-bf16')[0].message, /64 options/)
  question.options.pop()
  assert.deepEqual(api.validateDecisionRequest('state', [question], 'd1-2b-bf16'), [])
  await assert.rejects(api.requestDecisions(' ', [choice], new AbortController().signal, 'd1-2b-bf16'), /Some fields need attention/)
  assert.equal(calls.length, 0)
})

test('unavailable D1 surfaces the error without a Laya retry', async () => {
  const { api, calls } = await decisionsApi({ response: { ok: false, status: 503 }, error: { detail: 'D1 worker is unavailable.' } })
  await assert.rejects(api.requestDecisions('state', [choice], new AbortController().signal, 'd1-2b-bf16'), (error) => {
    assert.equal(error.title, 'Decision model unavailable')
    assert.equal(error.message, 'D1 worker is unavailable.')
    assert.equal(error.retryable, true)
    return true
  })
  assert.equal(calls.length, 1)
  assert.equal(calls[0].body.model, 'd1-2b-bf16')
})

test('D1 returns fractional Score and Noul probabilities without conversion', async () => {
  const questions = [score(2), { key: 'likely', type: 'Noul', instructions: 'Needs help.', trueWhen: 'Asks for help', falseWhen: 'Declines help' }]
  const answers = [
    { key: 'priority', type: 'Score', value: 0.7, confidence: 0.7, probabilities: { 0: 0.3, 1: 0.7 } },
    { key: 'likely', type: 'Noul', value: 0.8, confidence: 0.8, probabilities: null },
  ]
  const { api, calls } = await decisionsApi({ response: { ok: true }, data: { answers } })
  assert.deepEqual(await api.requestDecisions('state', questions, new AbortController().signal, 'd1-2b-bf16'), answers)
  assert.deepEqual(calls[0].body.questions, questions)
})
