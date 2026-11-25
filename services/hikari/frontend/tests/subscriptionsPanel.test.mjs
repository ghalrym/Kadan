import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(new URL('../src/components/SubscriptionsPanel.tsx', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
})
const modelsSource = await readFile(new URL('../src/data/models.ts', import.meta.url), 'utf8')
const models = ts.transpileModule(modelsSource, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText

function snapshot(capabilities = {}, connected = { claude: true, codex: true }) {
  return {
    capabilities: { language: 'local', image: 'local', ...capabilities },
    subscriptions: [
      { provider: 'claude', connected: connected.claude, model: 'claude-haiku-5-5' },
      { provider: 'codex', connected: connected.codex, model: 'gpt-6.1-terra' },
    ],
  }
}

function elements(node) {
  if (!node || typeof node !== 'object') return []
  return [node, ...[node.props?.children].flat(Infinity).flatMap(elements)]
}
function content(node) {
  if (Array.isArray(node)) return node.map(content).join(' ')
  if (node && typeof node === 'object') return content(node.props?.children)
  return node == null || typeof node === 'boolean' ? '' : String(node)
}
const alert = tree => elements(tree).filter(node => node.props?.role === 'alert').map(content).join(' ')
const group = (tree, label) => elements(tree).find(node => node.props?.['aria-label'] === `${label} provider`)
const capabilityRow = (tree, label) => elements(tree).find(node =>
  node.props?.className?.startsWith('subscription-capability') && content(node).includes(label))
const options = (tree, label) => elements(group(tree, label)).filter(node => node.type === 'button')
const option = (tree, label, name) => options(tree, label).find(node => content(node) === name)
const selected = (tree, label) => content(options(tree, label).find(node => node.props['aria-pressed']))

// Call SubscriptionsPanel directly with local hook state, JSX records and deferred API mocks.
// No React tree, DOM or service is mounted or invoked.
async function panel() {
  const slots = [], reads = [], choices = [], disconnects = []
  let cursor = 0, dirty = true, tree
  const request = (calls, extra) => new Promise((resolve, reject) => calls.push({ ...extra, resolve, reject }))
  const context = createContext({
    AbortController, Error,
    AbortSignal: { any: signals => AbortSignal.any(signals), timeout: () => new AbortController().signal },
    setInterval: () => ({}), clearInterval: () => {},
  })
  function module(exports) {
    return new SyntheticModule(Object.keys(exports), function () {
      for (const [name, value] of Object.entries(exports)) this.setExport(name, value)
    }, { context })
  }
  const react = module({
    useState(initial) {
      const index = cursor++
      slots[index] ??= { value: initial }
      return [slots[index].value, next => {
        const value = typeof next === 'function' ? next(slots[index].value) : next
        if (!Object.is(value, slots[index].value)) { slots[index].value = value; dirty = true }
      }]
    },
    useRef(initial) { const index = cursor++; return slots[index] ??= { current: initial } },
    useEffect(callback, deps) {
      const index = cursor++, previous = slots[index]
      if (!previous || deps.some((value, i) => !Object.is(value, previous.deps[i]))) {
        slots[index] = { deps }
        pending.push(() => { previous?.cleanup?.(); slots[index].cleanup = callback() })
      }
    },
  })
  const pending = []
  const jsx = (type, props) => ({ type, props })
  const api = module({
    accountNames: { claude: 'Claude', codex: 'ChatGPT' },
    providerNames: { local: 'Local', claude: 'Claude', codex: 'Codex' },
    imageProviderNames: { local: 'Local', codex: 'ChatGPT' },
    chooseImages: async () => {},
    modelNames: { 'claude-haiku-5-5': 'Claude Haiku 5.5', 'gpt-6.1-terra': 'GPT 6.1 Terra' },
    fetchSubscriptions: () => request(reads),
    chooseProvider: (capability, provider) => request(choices, { capability, provider }),
    disconnect: provider => request(disconnects, { provider }),
    cancelLogin: async () => {}, loginStatus: async () => {}, startLogin: async () => {}, submitLoginCode: async () => {},
  })
  const data = new SourceTextModule(models, { context })
  await data.link(() => {})
  const modules = {
    react, 'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }),
    '../api/subscriptions': api, '../data/models': data,
  }
  const component = new SourceTextModule(outputText, { context })
  await component.link(specifier => { assert.ok(modules[specifier], specifier); return modules[specifier] })
  await component.evaluate()
  async function flush() {
    for (let i = 0; i < 12; i++) {
      await Promise.resolve()
      if (dirty) {
        dirty = false; cursor = 0; tree = component.namespace.default()
        pending.splice(0).forEach(run => run())
      }
    }
    return tree
  }
  await flush()
  return {
    reads, choices, disconnects, flush,
    get tree() { return tree },
    async click(target) {
      assert.ok(target, 'Missing button')
      assert.ok(!target.props.disabled, `Disabled button: ${content(target)}`)
      target.props.onClick()
      await flush()
    },
  }
}

async function loaded(state) {
  const view = await panel()
  view.reads[0].resolve(state)
  await view.flush()
  return view
}

test('language offers Local Inkling 2 Mini, Claude and Codex without a separate vision toggle', async () => {
  const view = await loaded(snapshot())
  assert.equal(group(view.tree, 'Vision'), undefined)
  assert.deepEqual(options(view.tree, 'Language').map(content), ['Local', 'Claude', 'Codex'])
  assert.equal(selected(view.tree, 'Language'), 'Local')
  assert.match(content(capabilityRow(view.tree, 'Language')), /Inkling 2 Mini on this computer/)
  assert.match(content(capabilityRow(view.tree, 'Images')), /Qwen-Image on this computer/)
})

test('language subscription options wait for their sign-in', async () => {
  const view = await loaded(snapshot({}, { claude: false, codex: true }))
  const claude = option(view.tree, 'Language', 'Claude')
  assert.equal(claude.props.disabled, true)
  assert.equal(claude.props.title, 'Sign in to Claude first')
  assert.equal(option(view.tree, 'Language', 'Codex').props.disabled, false)
})

test('choosing Codex for language saves language and shows GPT 6.1 Terra', async () => {
  const view = await loaded(snapshot())
  await view.click(option(view.tree, 'Language', 'Codex'))
  assert.deepEqual(view.choices.map(({ capability, provider }) => [capability, provider]), [['language', 'codex']])
  view.choices[0].resolve(snapshot({ language: 'codex' }))
  await view.flush()
  assert.equal(selected(view.tree, 'Language'), 'Codex')
  assert.match(content(capabilityRow(view.tree, 'Language')), /GPT 6\.1 Terra/)
})

test('a failed language choice is reported and the selection stays', async () => {
  const view = await loaded(snapshot())
  await view.click(option(view.tree, 'Language', 'Claude'))
  view.choices[0].reject(new Error('Sign in to Claude again, then try once more.'))
  await view.flush()
  assert.equal(alert(view.tree), 'Sign in to Claude again, then try once more.')
  assert.equal(selected(view.tree, 'Language'), 'Local')
})

test('signing out shows language back on Local', async () => {
  const view = await loaded(snapshot({ language: 'claude' }))
  assert.equal(selected(view.tree, 'Language'), 'Claude')
  const signOut = elements(view.tree).find(node => node.type === 'button' && content(node) === 'Sign out')
  await view.click(signOut)
  assert.equal(view.disconnects[0].provider, 'claude')
  view.disconnects[0].resolve(snapshot({}, { claude: false, codex: true }))
  await view.flush()
  assert.equal(selected(view.tree, 'Language'), 'Local')
  assert.equal(option(view.tree, 'Language', 'Claude').props.disabled, true)
})
