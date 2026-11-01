import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(new URL('../src/pages/SettingsPage.tsx', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
})

const capabilities = [
  { id: 'lfm3-1.7b-bf16', label: 'Language', short: 'Language' },
  { id: 'whisper-medium', label: 'Speech to text', short: 'STT' },
]
function rows(status = 'not_downloaded', overrides = {}) {
  return capabilities.map(capability => ({ capability, model: {
    id: capability.id, display_name: `${capability.id} (experimental)`, repo_id: 'repo', status, license: 'lfm1.0',
    license_notice: capability.id.startsWith('lfm3') ? 'Review this license' : null,
    license_url: null, downloaded_bytes: 0, total_bytes: 0, estimated_bytes: 1e9, error: null, ...overrides,
  } }))
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

// Call SettingsPage directly with local hook state, JSX records and deferred API mocks.
// No React tree, DOM, native dialog or service is mounted or invoked.
async function page(componentName = 'default') {
  const slots = [], effects = [], timers = new Map(), reads = [], changes = [], timeouts = []
  let cursor = 0, dirty = true, tree, mounted = true, updatesAfterUnmount = 0
  function request(calls, signal, models) {
    return new Promise((resolve, reject) => calls.push({ signal, models, resolve, reject }))
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
    fetchModels: signal => request(reads, signal),
    downloadModels: (models, acknowledged, signal) => {
      const pending = request(changes, signal, Array.from(models))
      changes.at(-1).acknowledged = acknowledged
      return pending
    },
    downloadState: model => !model ? 'unavailable' : model.status === 'cancelled' ? 'failed'
      : model.status === 'downloading' && !model.total_bytes ? 'queued' : model.status,
    licenseNotices: models => models.map(({ model }) => model).filter(model => model?.license_notice && model.status !== 'complete'),
    modelsDownloading: models => models.some(({ model }) => model?.status === 'downloading'),
    modelsComplete: models => models.every(({ model }) => model?.status === 'complete'),
    modelsCanDownload: models => models.some(({ model }) => model && ['not_downloaded', 'failed', 'cancelled'].includes(model.status)),
    modelsProgress: () => 40,
  })
  const modules = {
    react, 'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }),
    '../api/settings': api,
    './SettingsPage.css': module({}),
  }
  const component = new SourceTextModule(outputText, { context })
  await component.link(specifier => { assert.ok(modules[specifier], specifier); return modules[specifier] })
  await component.evaluate()
  async function flush() {
    for (let i = 0; i < 12; i++) {
      await Promise.resolve()
      if (dirty && mounted) {
        dirty = false; cursor = 0; tree = component.namespace[componentName]()
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

async function loaded(status = 'not_downloaded', overrides) {
  const view = await page()
  view.reads[0].resolve(rows(status, overrides))
  await view.flush()
  return view
}

const modelRow = (tree, id) => elements(tree).find(node => node.type?.name === 'ModelRow' && node.props.capability.id === id)
const rowButton = (tree, id) => { const row = modelRow(tree, id); return elements(row.type(row.props)).find(node => node.type === 'button') }

async function downloadRow(view, id) {
  const target = rowButton(view.tree, id)
  assert.ok(target, `Missing row button: ${id}`)
  assert.equal(target.props.disabled, false)
  target.props.onClick()
  await view.flush()
}

async function acknowledge(view) {
  const checkbox = elements(modal(view.tree)).find(node => node.type === 'input')
  assert.equal(checkbox.props.checked, false)
  checkbox.props.onChange({ target: { checked: true } })
  await view.flush()
}

async function failedDownload() {
  const view = await loaded()
  await view.click('Download models')
  await acknowledge(view)
  await view.click('Accept and download', modal(view.tree))
  view.changes[0].reject(new Error('Queue unavailable'))
  await view.flush()
  return view
}

test('the page lists each capability without any tier choice', async () => {
  const view = await loaded()
  assert.match(content(view.tree), /0 of 2 downloaded/)
  assert.doesNotMatch(content(view.tree), /tier/i)
  const names = elements(view.tree).filter(node => node.type?.name === 'ModelRow').map(row => row.props.capability.short)
  assert.deepEqual(names, ['Language', 'STT'])
  assert.equal(button(view.tree, 'Download models').props.disabled, false)
  for (const { id } of capabilities) assert.equal(content(rowButton(view.tree, id)), 'Download')
  view.unmount()
})

test('a row download confirms and requests only that model', async () => {
  const view = await loaded()
  await downloadRow(view, capabilities[1].id)
  assert.match(content(modal(view.tree)), /This downloads\s+1\s+model\s*, about\s+1.0 GB/)
  assert.equal(modal(view.tree).props.title, 'Download Speech to text?')
  assert.doesNotMatch(content(modal(view.tree)), /Review this license/)
  await view.click('Download', modal(view.tree))
  assert.deepEqual(view.changes[0].models.map(row => row.capability.id), [capabilities[1].id])
  const next = rows()
  next[1].model.status = 'complete'
  view.changes[0].resolve(next)
  await view.flush()
  assert.match(content(view.tree), /1 of 2 downloaded/)
  assert.equal(rowButton(view.tree, capabilities[1].id), undefined)
  assert.equal(button(view.tree, 'Download models').props.disabled, false)
  view.unmount()
})

test('a row download shows only that model\'s license', async () => {
  const view = await loaded()
  await downloadRow(view, capabilities[0].id)
  assert.match(content(modal(view.tree)), /Review this license/)
  await acknowledge(view)
  await view.click('Accept and download', modal(view.tree))
  assert.deepEqual(view.changes[0].models.map(row => row.capability.id), [capabilities[0].id])
  view.unmount()
})

for (const status of ['complete', 'downloading', 'cancelling', 'unavailable']) {
  test(`${status} models offer no row download`, async () => {
    const view = await page()
    const models = rows()
    models[0].model = status === 'unavailable' ? null : { ...models[0].model, status }
    view.reads[0].resolve(models)
    await view.flush()
    assert.equal(rowButton(view.tree, capabilities[0].id), undefined)
    assert.equal(content(rowButton(view.tree, capabilities[1].id)), 'Download')
    view.unmount()
  })
}

test('the remaining models can start while another download is active', async () => {
  const view = await page()
  const models = rows()
  models[0].model.status = 'downloading'
  view.reads[0].resolve(models)
  await view.flush()
  await view.click('Download models')
  await view.click('Download', modal(view.tree))
  assert.deepEqual(view.changes[0].models.map(row => row.capability.id), [capabilities[1].id])
  view.unmount()
})

test('polling cannot change the batch or licenses after confirmation opens', async () => {
  const view = await page()
  const initial = rows()
  initial[1].model.status = 'downloading'
  view.reads[0].resolve(initial)
  await view.flush()
  await view.click('Download models')
  await acknowledge(view)
  const updated = rows()
  updated[0].model.license_notice = 'Changed license'
  updated[1].model.status = 'failed'
  updated[1].model.license_notice = 'Another license'
  view.reads.at(-1).resolve(updated)
  await view.flush()
  assert.match(content(modal(view.tree)), /Review this license/)
  assert.doesNotMatch(content(modal(view.tree)), /Changed license|Another license/)
  await view.click('Accept and download', modal(view.tree))
  assert.deepEqual(view.changes[0].models.map(row => row.capability.id), [capabilities[0].id])
  view.unmount()
})

test('an unavailable remaining model does not claim to be downloading', async () => {
  const view = await page()
  const models = rows('complete')
  models[1].model = null
  view.reads[0].resolve(models)
  await view.flush()
  assert.equal(button(view.tree, 'Unavailable').props.disabled, true)
  assert.equal(button(view.tree, 'Downloading…'), undefined)
  view.unmount()
})

test('opening the license dialog never starts or accepts downloads', async () => {
  const view = await loaded()
  await view.click('Download models')
  assert.match(content(modal(view.tree)), /Review this license/)
  assert.equal(button(modal(view.tree), 'Accept and download').props.disabled, true)
  await view.click('Cancel', modal(view.tree))
  assert.equal(modal(view.tree), undefined)
  await view.click('Download models')
  assert.equal(button(modal(view.tree), 'Accept and download').props.disabled, true)
  assert.equal(view.changes.length, 0)
  view.unmount()
})

test('explicit acknowledgement downloads the listed models and follows progress to completion', async () => {
  const view = await loaded()
  await view.click('Download models')
  await acknowledge(view)
  await view.click('Accept and download', modal(view.tree))
  assert.equal(view.changes.length, 1)
  assert.equal(view.changes[0].acknowledged, true)
  assert.deepEqual(view.changes[0].models.map(row => row.capability.id), capabilities.map(item => item.id))
  view.changes[0].resolve(rows('downloading', { total_bytes: 1e9 }))
  await view.flush()
  assert.equal(modal(view.tree), undefined)
  assert.match(content(view.tree), /Downloading…\s+40\s*%/)
  assert.equal(button(view.tree, 'Downloading…').props.disabled, true)
  view.reads.at(-1).resolve(rows('complete'))
  await view.flush()
  assert.match(content(view.tree), /All models downloaded/)
  assert.equal(button(view.tree, 'Download models'), undefined)
  view.unmount()
})

test('an uncertain download requires a refresh without replaying acceptance', async () => {
  const view = await failedDownload()
  assert.match(alert(modal(view.tree)), /Queue unavailable/)
  assert.match(alert(modal(view.tree)), /may have started/)
  assert.equal(button(modal(view.tree), 'Accept and download').props.disabled, true)
  await view.click('Refresh', modal(view.tree))
  view.reads.at(-1).resolve(rows('downloading'))
  await view.flush()
  assert.equal(view.changes.length, 1)
  assert.equal(modal(view.tree), undefined)
  assert.equal(alert(view.tree), '')
  view.unmount()
})

for (const status of ['failed', 'cancelled']) {
  test(`${status} downloads offer a resume and a row retry`, async () => {
    const view = await loaded(status)
    assert.ok(button(view.tree, 'Resume downloads'))
    assert.equal(content(rowButton(view.tree, capabilities[1].id)), 'Retry')
    await downloadRow(view, capabilities[1].id)
    await view.click('Download', modal(view.tree))
    assert.deepEqual(view.changes[0].models.map(row => row.capability.id), [capabilities[1].id])
    view.unmount()
  })
}

test('a timed-out download keeps the failure visible until refreshed', async () => {
  const view = await loaded()
  await view.click('Download models')
  await acknowledge(view)
  await view.click('Accept and download', modal(view.tree))
  view.timeouts[0].abort(new Error('Settings timed out'))
  view.changes[0].reject(view.changes[0].signal.reason)
  await view.flush()
  await view.click('Cancel', modal(view.tree))
  assert.match(alert(view.tree), /Settings timed out/)
  assert.equal(button(view.tree, 'Download models').props.disabled, true)
  assert.equal(rowButton(view.tree, capabilities[0].id).props.disabled, true)
  view.unmount()
})

test('downloads keep polling until every model completes', async () => {
  const view = await loaded('downloading', { total_bytes: 1e9 })
  view.reads.at(-1).resolve(rows('downloading', { total_bytes: 1e9, downloaded_bytes: 5e8 }))
  await view.flush()
  await view.poll()
  view.reads.at(-1).resolve(rows('complete'))
  await view.flush()
  assert.match(content(view.tree), /All models downloaded/)
  view.unmount()
})

for (const pending of ['poll', 'download']) {
  test(`unmount aborts a pending ${pending} and ignores its late result`, async () => {
    const view = await loaded(pending === 'poll' ? 'downloading' : 'not_downloaded', pending === 'poll' ? { total_bytes: 1e9 } : {})
    if (pending !== 'poll') {
      await view.click('Download models')
      await acknowledge(view)
      await view.click('Accept and download', modal(view.tree))
    }
    const request = pending === 'download' ? view.changes.at(-1) : view.reads.at(-1)
    view.unmount()
    assert.equal(request.signal.aborted, true)
    request.resolve(rows('complete'))
    await view.flush()
    assert.equal(view.updatesAfterUnmount, 0)
  })
}

for (const [label, overrides, total, shown] of [
  ['verified completion', { status: 'complete', downloaded_bytes: 3_100_000_000, total_bytes: 3_100_000_000, estimated_bytes: 9_000_000_000 }, '3.1 GB', '3.1 GB'],
  ['catalog estimate', { status: 'not_downloaded', downloaded_bytes: 0, total_bytes: 0, estimated_bytes: 9_000_000_000 }, '9.0 GB (estimated)', '9.0 GB (estimated)'],
  ['unknown size', { status: 'not_downloaded', downloaded_bytes: 0, total_bytes: 0, estimated_bytes: 0 }, 'Size unknown', 'Size unknown'],
  ['active observed total', { status: 'downloading', downloaded_bytes: 1_000_000_000, total_bytes: 2_000_000_000, estimated_bytes: 9_000_000_000 }, '2.0 GB', '1.0 GB / 2.0 GB'],
]) {
  test(`model rows and tooltips distinguish ${label}`, async () => {
    const view = await loaded(overrides.status, overrides)
    const row = elements(view.tree).find(node => node.type?.name === 'ModelRow')
    const rendered = row.type(row.props)
    const modelName = elements(rendered).find(node => node.type?.name === 'ModelName')
    const tooltip = elements(modelName.type(modelName.props)).find(node => node.props?.role === 'tooltip')
    assert.ok(content(tooltip).includes(total), content(tooltip))
    assert.doesNotMatch(content(tooltip), /experimental/)
    const size = elements(rendered).find(node => node.props?.className === 'mono faint model-row-size')
    assert.equal(content(size), shown)
    view.unmount()
  })
}
