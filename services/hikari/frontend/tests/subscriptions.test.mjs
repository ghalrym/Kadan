import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const sdkNames = ['cancelSubscriptionLogin', 'chooseCapabilityProvider', 'disconnectSubscription', 'getSubscriptionLogin',
  'getSubscriptions', 'startSubscriptionLogin', 'submitSubscriptionLoginCode']

// Replace only the generated HTTP boundary; exercise the actual API adapter.
async function subscriptionApi(response) {
  const calls = []
  const sdk = new SyntheticModule(sdkNames, function () {
    for (const name of sdkNames) this.setExport(name, async args => { calls.push({ name, args }); return response })
  })
  const source = await readFile(new URL('../src/api/subscriptions.ts', import.meta.url), 'utf8')
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link(specifier => {
    assert.equal(specifier, './generated/sdk.gen')
    return sdk
  })
  await module.evaluate()
  return { api: module.namespace, calls }
}

const snapshot = (image = 'local', codex = true) => ({
  subscriptions: [
    { provider: 'claude', connected: true, model: 'claude-haiku-5-5' },
    { provider: 'codex', connected: codex, model: 'gpt-6.1-terra' },
  ],
  capabilities: { language: 'local', image },
})

test('chooses the image provider separately from language', async () => {
  const { api, calls } = await subscriptionApi({ response: { ok: true }, data: snapshot('codex') })
  const signal = new AbortController().signal
  assert.deepEqual(await api.chooseImages('codex', signal), snapshot('codex'))
  assert.deepEqual(calls[0].args.path, { capability: 'image' })
  assert.deepEqual(calls[0].args.body, { provider: 'codex' })
  assert.equal(calls[0].args.signal, signal)
  await api.chooseLanguage('claude', signal)
  assert.deepEqual(calls[1].args.path, { capability: 'language' })
})

test('names ChatGPT as the only subscription image provider', async () => {
  const { api } = await subscriptionApi({})
  assert.deepEqual(api.imageProviderNames, { local: 'Local', codex: 'ChatGPT' })
})

test('shows the service explanation when an image provider is refused', async () => {
  const { api } = await subscriptionApi({ response: { ok: false, status: 409 }, error: { detail: 'Connect your Codex subscription first.' } })
  await assert.rejects(api.chooseImages('codex', new AbortController().signal), /Connect your Codex subscription first/)
})

// Render SubscriptionsPanel with hook stubs and JSX records; no DOM or service runs.
const panelSource = await readFile(new URL('../src/components/SubscriptionsPanel.tsx', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(panelSource, {
  compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
})
function elements(node) {
  if (!node || typeof node !== 'object') return []
  if (typeof node.type === 'function') return elements(node.type(node.props))
  return [node, ...[node.props?.children].flat(Infinity).flatMap(elements)]
}
function content(node) {
  if (Array.isArray(node)) return node.map(content).join(' ')
  if (node && typeof node === 'object') return content(node.props?.children)
  return node == null || typeof node === 'boolean' ? '' : String(node)
}

async function panel(state) {
  const slots = [], choices = []
  let cursor = 0, dirty = true, tree
  const context = createContext({ AbortController, Error, AbortSignal: { any: signals => signals[0], timeout: () => new AbortController().signal } })
  const module = exports => new SyntheticModule(Object.keys(exports), function () {
    for (const [name, value] of Object.entries(exports)) this.setExport(name, value)
  }, { context })
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
      const index = cursor++
      if (!slots[index] || deps.some((value, i) => !Object.is(value, slots[index].deps[i]))) {
        slots[index] = { deps }
        queueMicrotask(callback)
      }
    },
  })
  const jsx = (type, props) => ({ type, props })
  const api = module({
    accountNames: { claude: 'Claude', codex: 'ChatGPT' },
    providerNames: { local: 'Local', claude: 'Claude', codex: 'Codex' },
    imageProviderNames: { local: 'Local', codex: 'ChatGPT' },
    modelNames: {},
    fetchSubscriptions: async () => state,
    chooseImages: async provider => { choices.push(['image', provider]); return { ...state, capabilities: { ...state.capabilities, image: provider } } },
    chooseProvider: async (capability, provider) => { choices.push([capability, provider]); return state },
    cancelLogin: async () => {}, disconnect: async () => state, loginStatus: async () => {}, startLogin: async () => {}, submitLoginCode: async () => {},
  })
  const dataSource = await readFile(new URL('../src/data/models.ts', import.meta.url), 'utf8')
  const data = new SourceTextModule(ts.transpileModule(dataSource, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText, { context })
  await data.link(() => {})
  const modules = { '../data/models': data, react, 'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }), '../api/subscriptions': api }
  const component = new SourceTextModule(outputText, { context })
  await component.link(specifier => { assert.ok(modules[specifier], specifier); return modules[specifier] })
  await component.evaluate()
  async function flush() {
    for (let i = 0; i < 12; i++) {
      await new Promise(resolve => setTimeout(resolve))
      if (dirty) { dirty = false; cursor = 0; tree = component.namespace.default() }
    }
    return elements(tree)
  }
  return { choices, flush }
}

const group = (nodes, label) => nodes.find(node => node.props?.['aria-label'] === label)
const buttons = node => elements(node).filter(child => child.type === 'button')

test('offers Local and ChatGPT for images once Codex is signed in', async () => {
  const view = await panel(snapshot('local'))
  let nodes = await view.flush()
  const images = buttons(group(nodes, 'Image provider'))
  assert.deepEqual(images.map(content), ['Local', 'ChatGPT'])
  assert.equal(images[1].props.disabled, false)
  assert.equal(buttons(group(nodes, 'Language provider')).length, 3)
  images[1].props.onClick()
  nodes = await view.flush()
  assert.deepEqual(view.choices, [['image', 'codex']])
  assert.equal(buttons(group(nodes, 'Image provider'))[1].props['aria-pressed'], true)
  assert.match(content(nodes[0]), /Seeds are not available/)
  assert.match(content(nodes[0]), /Editing images and pets still use this computer/)
})

test('keeps ChatGPT images unavailable until Codex is signed in', async () => {
  const view = await panel(snapshot('local', false))
  const nodes = await view.flush()
  const chatgpt = buttons(group(nodes, 'Image provider'))[1]
  assert.equal(chatgpt.props.disabled, true)
  assert.equal(chatgpt.props.title, 'Sign in to Codex first')
  assert.match(content(nodes[0]), /Claude cannot create images/)
})
