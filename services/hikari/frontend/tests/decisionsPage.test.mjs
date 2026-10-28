import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(new URL('../src/pages/DecisionsPage.tsx', import.meta.url), 'utf8')
const apiSource = await readFile(new URL('../src/api/decisions.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX } })
const d1 = (status = 'complete') => ({ id: 'd1-2b-bf16', status, inference_available: true })
const question = { key: 'intent', type: 'Choice', instructions: 'Pick intent.', options: [{ key: 'help', description: '' }] }
const answer = { key: 'intent', type: 'Choice', value: 'help' }
const elements = (node) => !node || typeof node !== 'object' ? [] : [node, ...[node.props?.children].flat(Infinity).flatMap(elements)]
const content = (node) => Array.isArray(node) ? node.map(content).join('') : node && typeof node === 'object' ? content(node.props?.children) : node == null || typeof node === 'boolean' ? '' : String(node)

// Local hooks, JSX records and deferred API calls only; no DOM or services.
async function page(models = [d1()]) {
  const slots = [], effects = [], reads = [], calls = [], timers = new Set()
  let cursor = 0, dirty = true, tree, mounted = true, lateUpdates = 0
  const context = createContext({ AbortController, DOMException, Error, Date,
    setInterval: () => { const id = {}; timers.add(id); return id },
    clearInterval: (id) => timers.delete(id),
  })
  const module = (exports) => new SyntheticModule(Object.keys(exports), function () {
    for (const [name, value] of Object.entries(exports)) this.setExport(name, value)
  }, { context })
  const defer = (list, args) => new Promise((resolve, reject) => list.push({ ...args, resolve, reject }))
  const sdk = module({ evaluateDecisions: (args) => defer(calls, args) })
  const api = new SourceTextModule(stripTypeScriptTypes(apiSource), { context })
  await api.link(() => sdk)
  const jsx = (type, props) => ({ type, props })
  const modules = {
    react: module({
      useState(initial) {
        const index = cursor++
        slots[index] ??= { value: typeof initial === 'function' ? initial() : initial }
        return [slots[index].value, (next) => {
          if (!mounted) lateUpdates++
          const value = typeof next === 'function' ? next(slots[index].value) : next
          if (!Object.is(value, slots[index].value)) { slots[index].value = value; dirty = true }
        }]
      },
      useRef(initial) { const index = cursor++; return slots[index] ??= { current: initial } },
      useEffect(callback, deps) {
        const index = cursor++, previous = slots[index]
        if (!previous || deps.some((value, i) => !Object.is(value, previous.deps[i]))) {
          slots[index] = { deps, cleanup: previous?.cleanup }
          effects.push(() => { slots[index].cleanup?.(); slots[index].cleanup = callback() })
        }
      },
    }),
    'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }),
    '../components/Controls': module({ Panel: () => {}, SectionHeading: () => {} }),
    '../components/DecisionQuestionCard': module({ default: function DecisionQuestionCard() {} }),
    '../components/DecisionAnswerCard': module({ default: function DecisionAnswerCard() {} }),
    '../api/decisions': api,
    '../api/settings': module({ fetchExperimentalModels: (signal) => defer(reads, { signal }) }),
  }
  const component = new SourceTextModule(outputText, { context })
  await component.link((name) => { assert.ok(modules[name], name); return modules[name] })
  await component.evaluate()
  async function flush() {
    for (let i = 0; i < 12; i++) {
      await Promise.resolve()
      if (dirty && mounted) {
        dirty = false; cursor = 0; tree = component.namespace.default()
        effects.splice(0).forEach((run) => run())
      }
    }
  }
  await flush()
  if (models) { reads[0].resolve(models); await flush() }
  const find = (predicate) => elements(tree).find(predicate)
  return {
    calls, reads, flush,
    get tree() { return tree },
    get lateUpdates() { return lateUpdates },
    async select(value) { find((node) => node.type === 'select').props.onChange({ target: { value } }); await flush() },
    async fill(state = 'Customer asks for help.') {
      find((node) => node.type === 'textarea').props.onChange({ target: { value: state } })
      await flush()
      await this.click('+ Choice')
      find((node) => node.type?.name === 'DecisionQuestionCard').props.onChange(question)
      await flush()
    },
    async submit() { find((node) => node.type === 'form').props.onSubmit({ preventDefault() {} }); await flush() },
    async click(label) {
      const button = find((node) => node.type === 'button' && content(node) === label)
      assert.ok(button, label); assert.ok(!button.props.disabled, label)
      button.props.onClick(); await flush()
    },
    unmount() { mounted = false; slots.forEach((slot) => slot.cleanup?.()); assert.equal(timers.size, 0) },
  }
}

test('Laya stays the default with D1 connected and no download is initiated', async () => {
  const view = await page()
  assert.equal(elements(view.tree).find((node) => node.type === 'select').props.value, 'laya')
  assert.match(content(view.tree), /D1 2B BF16/)
  await view.fill(); await view.submit()
  assert.equal(view.calls[0].body.model, undefined)
  assert.match(content(view.tree), /Laya on CPU/)
  view.unmount()
})

test('D1 is offered only after discovery and refresh can discover a new worker', async () => {
  const view = await page([])
  assert.equal(elements(view.tree).some((node) => node.type === 'option' && node.props.value === 'd1-2b-bf16'), false)
  await view.select('d1-2b-bf16')
  assert.equal(elements(view.tree).find((node) => node.type === 'select').props.value, 'laya')
  await view.click('Refresh models')
  view.reads.at(-1).resolve([d1()]); await view.flush()
  assert.match(content(view.tree), /D1 2B BF16/)
  view.unmount()
})

test('D1 without a completed download points to Settings and cannot evaluate', async () => {
  const view = await page([d1('not_downloaded')])
  await view.select('d1-2b-bf16'); await view.fill(); await view.submit()
  assert.match(content(view.tree), /Review its license and download/)
  assert.equal(elements(view.tree).find((node) => node.type === 'a').props.href, '/settings')
  assert.equal(elements(view.tree).find((node) => node.type === 'button' && node.props.type === 'submit').props.disabled, true)
  assert.equal(view.calls.length, 0)
  view.unmount()
})

test('D1 accepts long state and switching models marks prior answers outdated', async () => {
  const view = await page()
  await view.select('d1-2b-bf16')
  const state = 'x'.repeat(17000)
  await view.fill(state); await view.submit()
  assert.equal(view.calls[0].body.state, state)
  assert.equal(view.calls[0].body.model, 'd1-2b-bf16')
  view.calls[0].resolve({ response: { ok: true }, data: { answers: [answer] } }); await view.flush()
  await view.select('laya')
  assert.match(content(view.tree), /Inputs or model changed/)
  assert.equal(elements(view.tree).find((node) => node.type?.name === 'DecisionAnswerCard').props.stale, true)
  await view.click('Re-evaluate')
  assert.equal(view.calls.length, 1)
  assert.match(content(view.tree), /8,000 characters/)
  view.unmount()
})

test('cancelled D1 ignores late answers while the current model evaluates', async () => {
  const view = await page()
  await view.select('d1-2b-bf16'); await view.fill(); await view.submit()
  // A stale change handler cannot switch the model while its request is active.
  await view.select('laya')
  assert.equal(elements(view.tree).find((node) => node.type === 'select').props.value, 'd1-2b-bf16')
  await view.click('Cancel evaluation')
  assert.equal(view.calls[0].signal.aborted, true)
  await view.select('laya'); await view.submit()
  view.calls[0].resolve({ response: { ok: true }, data: { answers: [answer] } }); await view.flush()
  assert.match(content(view.tree), /Evaluating 1 question with Laya/)
  view.calls[1].resolve({ response: { ok: true }, data: { answers: [answer] } }); await view.flush()
  assert.equal(elements(view.tree).find((node) => node.type?.name === 'DecisionAnswerCard').props.stale, false)
  view.unmount()
})

test('D1 failure stays visible and explicit retry uses the current model', async () => {
  const view = await page()
  await view.select('d1-2b-bf16'); await view.fill(); await view.submit()
  view.calls[0].resolve({ response: { ok: false, status: 503 }, error: { detail: 'D1 worker unavailable.' } })
  await view.flush()
  assert.match(content(view.tree), /D1 worker unavailable/)
  assert.equal(view.calls.length, 1)
  await view.select('laya')
  await view.click('Retry')
  assert.equal(view.calls.length, 2)
  assert.equal(view.calls[1].body.model, undefined)
  view.unmount()
})

test('leaving Decisions aborts discovery and evaluation and ignores their late results', async () => {
  const view = await page()
  await view.select('d1-2b-bf16'); await view.fill(); await view.submit()
  view.unmount()
  assert.equal(view.reads[0].signal.aborted, true)
  assert.equal(view.calls[0].signal.aborted, true)
  view.calls[0].resolve({ response: { ok: true }, data: { answers: [answer] } }); await view.flush()
  assert.equal(view.lateUpdates, 0)
  const discovery = await page(null)
  discovery.unmount()
  discovery.reads[0].resolve([d1()]); await discovery.flush()
  assert.equal(discovery.lateUpdates, 0)
})
