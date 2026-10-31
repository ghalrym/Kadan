import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

async function sourceModule(path, jsx = false) {
  const source = await readFile(new URL(path, import.meta.url), 'utf8')
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: jsx ? ts.JsxEmit.ReactJSX : undefined },
  })
  return new SourceTextModule(outputText)
}
function mock(exports) {
  return new SyntheticModule(Object.keys(exports), function () {
    for (const [name, value] of Object.entries(exports)) this.setExport(name, value)
  })
}
const score = count => ({
  type: 'Score', key: 'urgency', instructions: 'How urgent is this?',
  levels: Array.from({ length: count }, (_, index) => `Level ${index}`),
})
async function api(evaluate = async () => assert.fail('Invalid requests must not reach the API')) {
  const module = await sourceModule('../src/api/decisions.ts')
  await module.link(specifier => {
    assert.equal(specifier, './generated/sdk.gen')
    return mock({ evaluateDecisions: evaluate })
  })
  await module.evaluate()
  return module.namespace
}

test('Score accepts two through ten levels and rejects unsupported sizes without a request', async () => {
  const decisions = await api()
  for (const count of [2, 10])
    assert.deepEqual(decisions.validateDecisionRequest('A support message', [score(count)]), [])
  for (const count of [0, 1, 11]) {
    const issues = decisions.validateDecisionRequest('A support message', [score(count)])
    assert.deepEqual(issues, [{ question: 0, message: 'Score requires between 2 and 10 rubric levels.' }])
    await assert.rejects(
      decisions.requestDecisions('A support message', [score(count)], new AbortController().signal),
      error => error instanceof decisions.DecisionError && error.issues[0].question === 0 && !error.retryable,
    )
  }
})

test('ordinary Decisions sends no model selector and preserves fractional Score answers and abort signal', async () => {
  const calls = [], questions = [score(10)], signal = new AbortController().signal
  const answers = [{ type: 'Score', key: 'urgency', value: 3.25 }]
  const decisions = await api(async request => {
    calls.push(request)
    return { response: { ok: true }, data: { answers } }
  })
  assert.deepEqual(await decisions.requestDecisions('A support message', questions, signal), answers)
  assert.deepEqual(calls, [{ body: { state: 'A support message', questions }, signal }])
})

test('server Score validation remains attached to its question card', async () => {
  const decisions = await api(async () => ({
    response: { ok: false, status: 422 },
    error: { detail: [{ loc: ['body', 'questions', 0, 'Score', 'levels'], msg: 'Score requires between 2 and 10 rubric levels.' }] },
  }))
  await assert.rejects(decisions.requestDecisions('A support message', [score(2)], new AbortController().signal), error => {
    assert.equal(error.title, 'Check your inputs')
    assert.deepEqual(error.issues, [{ question: 0, message: 'Levels: Score requires between 2 and 10 rubric levels.' }])
    return true
  })
})

function elements(node) {
  if (!node || typeof node !== 'object') return []
  return [node, ...[node.props?.children].flat(Infinity).flatMap(elements)]
}
function content(node) {
  if (Array.isArray(node)) return node.map(content).join('')
  if (node && typeof node === 'object') return content(node.props?.children)
  return node == null || typeof node === 'boolean' ? '' : String(node)
}
async function questionCard(question) {
  const changes = []
  const jsx = (type, props) => ({ type, props })
  const modules = {
    react: mock({ useId: () => 'question' }),
    'react/jsx-runtime': mock({ jsx, jsxs: jsx }),
  }
  const component = await sourceModule('../src/components/DecisionQuestionCard.tsx', true)
  await component.link(specifier => { assert.ok(modules[specifier], specifier); return modules[specifier] })
  await component.evaluate()
  const tree = component.namespace.default({ question, index: 0, onChange: value => changes.push(value), onRemove() {} })
  return {
    changes,
    add: elements(tree).find(node => node.type === 'button' && content(node).includes('Add level')),
    remove: elements(tree).filter(node => node.type === 'button' && node.props['aria-label']?.startsWith('Remove level')),
  }
}

test('Score controls keep the rubric between two and ten ordered levels', async () => {
  const minimum = await questionCard(score(2))
  assert.equal(minimum.remove.length, 2)
  assert.equal(minimum.remove.every(button => button.props.disabled), true)
  assert.equal(minimum.add.props.disabled, false)
  minimum.add.props.onClick()
  assert.deepEqual(minimum.changes[0].levels, ['Level 0', 'Level 1', ''])

  const maximum = await questionCard(score(10))
  assert.equal(maximum.add.props.disabled, true)
  assert.equal(maximum.remove.some(button => button.props.disabled), false)
  maximum.remove[4].props.onClick()
  assert.deepEqual(maximum.changes[0].levels, score(10).levels.filter((_, index) => index !== 4))
})
