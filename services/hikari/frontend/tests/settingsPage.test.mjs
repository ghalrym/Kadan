import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(new URL('../src/pages/SettingsPage.tsx', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
})

function saved(tier = 'go', status = 'failed') {
  return { tier, tiers: ['go', 'pro'].map(id => ({
    id, name: id === 'go' ? 'Go' : 'Pro', vram_gb: 8, ram_gb: 32,
    models: [{ modality: 'llm', model: {
      id: `${id}-model`, status, license_notice: 'Review this license',
    } }],
  })) }
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
const modal = tree => elements(tree).find(node => node.type?.name === 'Modal')
const alert = tree => elements(tree).filter(node => node.props?.role === 'alert').map(content).join(' ')
const button = (tree, label) => elements(tree).find(node => node.type === 'button' && content(node).includes(label))
const card = (tree, id) => elements(tree).find(node => node.type?.name === 'TierCard' && node.props.tier.id === id)

// Call SettingsPage directly with local hook state, JSX records and deferred API mocks.
// No React tree, DOM, native dialog or service is mounted or invoked.
async function page() {
  const slots = [], effects = [], timers = new Map(), reads = [], changes = [], timeouts = []
  let cursor = 0, dirty = true, tree, mounted = true, updatesAfterUnmount = 0
  function request(calls, signal, tier) {
    return new Promise((resolve, reject) => calls.push({ signal, tier, resolve, reject }))
  }
  const context = createContext({
    AbortController, Error,
    AbortSignal: {
      any: signals => AbortSignal.any(signals),
      timeout: () => { const controller = new AbortController(); timeouts.push(controller); return controller.signal },
    },
    setTimeout: callback => { const id = {}; timers.set(id, callback); return id },
    clearTimeout: id => timers.delete(id),
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
        if (!mounted) updatesAfterUnmount++
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
  })
  const jsx = (type, props) => ({ type, props })
  const api = module({
    fetchSettings: signal => request(reads, signal),
    chooseTier: (tier, signal) => request(changes, signal, tier),
    downloadState: model => model.status,
    licenseNotices: tier => tier.models.map(({ model }) => model),
    tierDownloading: tier => tier.models.some(({ model }) => model.status === 'downloading'),
    tierComplete: tier => tier.models.every(({ model }) => model.status === 'complete'),
    tierCanDownload: tier => tier.models.some(({ model }) => model.status === 'failed'),
    tierProgress: () => 0,
  })
  const modules = {
    react, 'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }),
    '../api/settings': api,
    '../data/tiers': module({ tierCopy: { go: {}, pro: {} }, modalityLabels: {} }),
    './SettingsPage.css': module({}),
  }
  const component = new SourceTextModule(outputText, { context })
  await component.link(specifier => { assert.ok(modules[specifier], specifier); return modules[specifier] })
  await component.evaluate()
  async function flush() {
    for (let i = 0; i < 12; i++) {
      await Promise.resolve()
      if (dirty && mounted) {
        dirty = false; cursor = 0; tree = component.namespace.default()
        effects.splice(0).forEach(run => run())
      }
    }
    return tree
  }
  await flush()
  return {
    reads, changes, timeouts, flush,
    get tree() { return tree },
    get updatesAfterUnmount() { return updatesAfterUnmount },
    async click(label, root = tree) {
      const target = button(root, label)
      assert.ok(target, `Missing button: ${label}`)
      assert.ok(!target.props.disabled, `Disabled button: ${label}`)
      target.props.onClick()
      await flush()
    },
    async open() { card(tree, 'go').props.showProgress(); await flush() },
    async poll() {
      const next = timers.entries().next().value
      assert.ok(next, 'Expected scheduled poll')
      timers.delete(next[0]); next[1](); await flush()
    },
    unmount() {
      mounted = false
      slots.forEach(slot => slot.cleanup?.())
      assert.equal(timers.size, 0)
    },
  }
}

async function progress() {
  const view = await page()
  view.reads[0].resolve(saved())
  await view.flush()
  await view.open()
  view.reads[1].resolve(saved())
  await view.flush()
  return view
}

async function failedResume() {
  const view = await progress()
  await view.click('Resume downloads')
  view.changes[0].reject(new Error('Queue unavailable'))
  await view.flush()
  return view
}

test('resume failure stays actionable inside the open modal through a successful poll and after close', async () => {
  const view = await failedResume()
  assert.match(alert(modal(view.tree)), /Queue unavailable/)
  assert.ok(button(modal(view.tree), 'Refresh saved state'))
  await view.poll()
  view.reads.at(-1).resolve(saved())
  await view.flush()
  assert.match(alert(modal(view.tree)), /Queue unavailable/)
  await view.click('Done')
  assert.equal(modal(view.tree), undefined)
  assert.match(alert(view.tree), /Queue unavailable/)
  view.unmount()
})

test('a successful read after closing the dialog cannot erase the failed action', async () => {
  const view = await failedResume()
  await view.click('Done')
  assert.match(alert(view.tree), /Queue unavailable/)
  view.reads.at(-1).resolve(saved())
  await view.flush()
  assert.match(alert(view.tree), /Queue unavailable/)
  view.unmount()
})

test('refresh failure keeps mutation locked; recovery reads saved state without replaying the change', async () => {
  const view = await failedResume()
  assert.equal(button(view.tree, 'Resume downloads').props.disabled, true)
  assert.equal(card(view.tree, 'pro').props.disabled, true)
  await view.click('Refresh saved state', modal(view.tree))
  assert.equal(button(view.tree, 'Refreshing').props.disabled, true)
  view.reads.at(-1).reject(new Error('Read unavailable'))
  await view.flush()
  assert.match(alert(modal(view.tree)), /Queue unavailable/)
  assert.match(alert(modal(view.tree)), /Read unavailable/)
  await view.click('Refresh saved state', modal(view.tree))
  view.reads.at(-1).resolve(saved('pro'))
  await view.flush()
  assert.match(modal(view.tree).props.title, /Pro/)
  assert.match(alert(modal(view.tree)), /Queue unavailable/)
  assert.doesNotMatch(alert(modal(view.tree)), /Read unavailable/)
  assert.equal(view.changes.length, 1)
  assert.equal(button(view.tree, 'Resume downloads').props.disabled, false)
  await view.click('Resume downloads')
  assert.equal(view.changes[1].tier, 'pro')
  assert.equal(alert(view.tree), '')
  view.changes[1].resolve(saved('pro', 'downloading'))
  await view.flush()
  assert.equal(card(view.tree, 'go').props.disabled, true)
  await view.click('Continue in background')
  assert.equal(view.changes[1].signal.aborted, false)
  view.unmount()
})

test('a timed-out tier change refreshes partial saved state before a fresh license confirmation', async () => {
  const view = await page()
  view.reads[0].resolve(saved())
  await view.flush()
  card(view.tree, 'pro').props.choose()
  await view.flush()
  assert.match(content(modal(view.tree)), /Review this license/)
  await view.click('Accept and download')
  view.timeouts[0].abort(new Error('Settings timed out'))
  view.changes[0].reject(view.changes[0].signal.reason)
  await view.flush()
  assert.match(alert(view.tree), /Settings timed out/)
  await view.click('Refresh saved state')
  view.reads.at(-1).resolve(saved('pro'))
  await view.flush()
  assert.equal(card(view.tree, 'pro').props.current, true)
  assert.equal(view.changes.length, 1)
  card(view.tree, 'go').props.choose()
  await view.flush()
  assert.match(content(modal(view.tree)), /Review this license/)
  await view.click('Accept and download')
  assert.equal(view.changes[1].tier, 'go')
  view.unmount()
})

test('recoverable read errors clear on a successful poll without clearing action failures', async () => {
  const view = await failedResume()
  await view.poll()
  view.reads.at(-1).reject(new Error('Poll unavailable'))
  await view.flush()
  assert.match(alert(modal(view.tree)), /Poll unavailable/)
  await view.poll()
  view.reads.at(-1).resolve(saved())
  await view.flush()
  assert.doesNotMatch(alert(modal(view.tree)), /Poll unavailable/)
  assert.match(alert(modal(view.tree)), /Queue unavailable/)
  view.unmount()
})

test('pending actions ignore older reads, block duplicate submissions and pause new reads', async () => {
  const view = await progress()
  await view.poll()
  const read = view.reads.at(-1)
  await view.click('Resume downloads')
  read.resolve(saved('pro'))
  await view.flush()
  assert.equal(card(view.tree, 'go').props.current, true)
  assert.equal(card(view.tree, 'pro').props.disabled, true)
  button(view.tree, 'Resume downloads').props.onClick()
  assert.equal(view.changes.length, 1)
  await view.poll()
  assert.equal(view.reads.at(-1), read)
  await view.click('Done')
  assert.equal(view.changes[0].signal.aborted, false)
  view.changes[0].resolve(saved('pro', 'downloading'))
  await view.flush()
  assert.equal(card(view.tree, 'pro').props.current, true)
  assert.equal(card(view.tree, 'go').props.disabled, true)
  view.unmount()
})

test('a poll started before explicit recovery cannot overwrite refreshed saved settings', async () => {
  const view = await failedResume()
  await view.poll()
  const read = view.reads.at(-1)
  await view.click('Refresh saved state')
  view.reads.at(-1).resolve(saved('pro'))
  await view.flush()
  read.resolve(saved('go'))
  await view.flush()
  assert.equal(card(view.tree, 'pro').props.current, true)
  assert.match(alert(modal(view.tree)), /Queue unavailable/)
  assert.equal(view.changes.length, 1)
  view.unmount()
})

for (const obsolete of ['resolve', 'reject']) {
  test(`an obsolete read cannot ${obsolete} over a completed action snapshot`, async () => {
    const view = await progress()
    await view.poll()
    const read = view.reads.at(-1)
    await view.click('Resume downloads')
    assert.equal(button(view.tree, 'Resume downloads').props.disabled, true)
    button(view.tree, 'Resume downloads').props.onClick()
    assert.equal(view.changes.length, 1)
    view.changes[0].resolve(saved('pro'))
    await view.flush()
    read[obsolete](obsolete === 'resolve' ? saved('go') : new Error('Obsolete read failure'))
    await view.flush()
    assert.equal(card(view.tree, 'pro').props.current, true)
    assert.equal(alert(view.tree), '')
    view.unmount()
  })
}

for (const pending of ['poll', 'mutation', 'refresh']) {
  test(`unmount aborts a pending ${pending} and ignores its late result`, async () => {
    const view = pending === 'refresh' ? await failedResume() : await progress()
    if (pending === 'poll') await view.poll()
    if (pending === 'mutation') await view.click('Resume downloads')
    if (pending === 'refresh') await view.click('Refresh saved state')
    const request = pending === 'mutation' ? view.changes.at(-1) : view.reads.at(-1)
    view.unmount()
    assert.equal(request.signal.aborted, true)
    request.resolve(saved('pro'))
    await view.flush()
    assert.equal(view.updatesAfterUnmount, 0)
  })
}
