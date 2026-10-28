import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(
  new URL('../src/pages/SettingsPage.tsx', import.meta.url),
  'utf8',
)
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
})

function elements(node) {
  if (!node || typeof node !== 'object') return []
  return [node, ...[node.props?.children].flat(Infinity).flatMap(elements)]
}
function content(node) {
  if (Array.isArray(node)) return node.map(content).join(' ')
  if (node && typeof node === 'object') return content(node.props?.children)
  return node == null || typeof node === 'boolean' ? '' : String(node)
}
const modal = (tree) =>
  elements(tree).find((node) => node.type?.name === 'Modal')
const alert = (tree) =>
  elements(tree)
    .filter((node) => node.props?.role === 'alert')
    .map(content)
    .join(' ')
const button = (tree, label) =>
  elements(tree).find(
    (node) => node.type === 'button' && content(node).includes(label),
  )

// Call SettingsPage directly with local hook state, JSX records and deferred API mocks.
// No React tree, DOM, native dialog or service is mounted or invoked.
async function page(componentName = 'ModelDownloads') {
  const slots = [],
    effects = [],
    timers = new Map(),
    reads = [],
    changes = [],
    timeouts = []
  let cursor = 0,
    dirty = true,
    tree,
    mounted = true,
    updatesAfterUnmount = 0
  function request(calls, signal, modelId) {
    return new Promise((resolve, reject) =>
      calls.push({ signal, modelId, resolve, reject }),
    )
  }
  const context = createContext({
    AbortController,
    Error,
    AbortSignal: {
      any: (signals) => AbortSignal.any(signals),
      timeout: () => {
        const controller = new AbortController()
        timeouts.push(controller)
        return controller.signal
      },
    },
    setTimeout: (callback) => {
      const id = {}
      timers.set(id, callback)
      return id
    },
    clearTimeout: (id) => timers.delete(id),
  })
  function module(exports) {
    return new SyntheticModule(
      Object.keys(exports),
      function () {
        for (const [name, value] of Object.entries(exports))
          this.setExport(name, value)
      },
      { context },
    )
  }
  const react = module({
    useState(initial) {
      const index = cursor++
      slots[index] ??= { value: initial }
      return [
        slots[index].value,
        (next) => {
          if (!mounted) updatesAfterUnmount++
          const value =
            typeof next === 'function' ? next(slots[index].value) : next
          if (!Object.is(value, slots[index].value)) {
            slots[index].value = value
            dirty = true
          }
        },
      ]
    },
    useRef(initial) {
      const index = cursor++
      return (slots[index] ??= { current: initial })
    },
    useEffect(callback, deps) {
      const index = cursor++,
        previous = slots[index]
      if (
        !previous ||
        deps.some((value, i) => !Object.is(value, previous.deps[i]))
      ) {
        slots[index] = { deps, cleanup: previous?.cleanup }
        effects.push(() => {
          slots[index].cleanup?.()
          slots[index].cleanup = callback()
        })
      }
    },
  })
  const jsx = (type, props) => ({ type, props })
  function change(operation, id, signal, fields = {}) {
    const pending = request(changes, signal, id)
    Object.assign(changes.at(-1), { operation, ...fields })
    return pending
  }
  const api = module({
    fetchSettings: (signal) => request(reads, signal),
    downloadModel: (id, acknowledged, signal) =>
      change('download', id, signal, { acknowledged }),
    cancelModelDownload: (id, signal) => change('cancel', id, signal),
    configureModelContext: (id, context, signal) =>
      change('context', id, signal, { context }),
    configuredModels: (saved) =>
      Object.entries(saved.defaults).map(([feature, id]) => ({
        id,
        uses: [feature],
        checkpoint: saved.models.find((model) => model.id === id) ?? null,
      })),
    downloadState: (model) =>
      model
        ? model.status === 'cancelled'
          ? 'failed'
          : model.status
        : 'unassigned',
  })
  const modules = {
    react,
    'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }),
    '../api/settings': api,
    './SettingsPage.css': module({}),
  }
  const component = new SourceTextModule(outputText, { context })
  await component.link((specifier) => {
    assert.ok(modules[specifier], specifier)
    return modules[specifier]
  })
  await component.evaluate()
  async function flush() {
    for (let i = 0; i < 12; i++) {
      await Promise.resolve()
      if (dirty && mounted) {
        dirty = false
        cursor = 0
        tree = component.namespace[componentName]()
        effects.splice(0).forEach((run) => run())
      }
    }
    return tree
  }
  await flush()
  return {
    reads,
    changes,
    timeouts,
    flush,
    get tree() {
      return tree
    },
    get updatesAfterUnmount() {
      return updatesAfterUnmount
    },
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
      timers.delete(next[0])
      next[1]()
      await flush()
    },
    unmount() {
      mounted = false
      slots.forEach((slot) => slot.cleanup?.())
      assert.equal(timers.size, 0)
    },
  }
}

const textModel = (status = 'not_downloaded') => ({
  id: 'lfm3-1.7b-bf16',
  kind: 'llm',
  display_name: 'LFM3 text',
  status,
  license: 'lfm1.0',
  license_notice: 'Review this text model license.',
  license_url: 'https://example.invalid/text-license',
  downloaded_bytes: 0,
  total_bytes: 100,
  estimated_bytes: 100,
  context_limit: 8192,
  architecture_context_limit: 262144,
  inference_available: true,
})
const saved = (models = [textModel()]) => ({
  defaults: {
    llm: 'lfm3-1.7b-bf16',
    image: 'qwen-image-2.1-turbo',
    speech: 'lfm3-audio-2b-bf16',
  },
  models,
  video_defaults: { duration: 5, resolution: '480p' },
  whisper_formatting: true,
})
async function loaded(models) {
  const view = await page()
  view.reads[0].resolve(saved(models))
  await view.flush()
  return view
}
async function acknowledge(view) {
  const checkbox = elements(modal(view.tree)).find(
    (node) => node.type === 'input' && node.props.type === 'checkbox',
  )
  assert.equal(checkbox.props.checked, false)
  checkbox.props.onChange({ target: { checked: true } })
  await view.flush()
}

test('fixed models show missing workers honestly without tier controls', async () => {
  const view = await loaded()
  assert.match(content(view.tree), /lfm3-audio-2b-bf16/)
  assert.match(content(view.tree), /Connect this model’s worker/)
  assert.doesNotMatch(
    content(view.tree),
    /Choose your tier|Choose Go|Experimental models/,
  )
  assert.equal(
    elements(view.tree).filter((node) =>
      node.props?.['aria-label']?.startsWith('Download '),
    ).length,
    1,
  )
  view.unmount()
})

test('license dialog cannot download before fresh explicit acknowledgement', async () => {
  const view = await loaded()
  await view.click('Download')
  assert.match(content(modal(view.tree)), /Review this text model license/)
  assert.equal(
    button(modal(view.tree), 'Accept and download').props.disabled,
    true,
  )
  button(modal(view.tree), 'Accept and download').props.onClick()
  await view.flush()
  assert.equal(view.changes.length, 0)
  await acknowledge(view)
  await view.click('Cancel', modal(view.tree))
  await view.click('Download')
  assert.equal(
    button(modal(view.tree), 'Accept and download').props.disabled,
    true,
  )
  view.unmount()
})

test('download starts only selected checkpoint and follows actual progress', async () => {
  const view = await loaded()
  await view.click('Download')
  await acknowledge(view)
  await view.click('Accept and download', modal(view.tree))
  assert.equal(view.changes[0].modelId, 'lfm3-1.7b-bf16')
  assert.equal(view.changes[0].acknowledged, true)
  view.changes[0].resolve([textModel('downloading')])
  await view.flush()
  assert.equal(modal(view.tree), undefined)
  assert.ok(button(view.tree, 'Cancel download'))
  view.reads.at(-1).resolve(saved([textModel('complete')]))
  await view.flush()
  assert.equal(button(view.tree, 'Downloaded').props.disabled, true)
  view.unmount()
})

test('cancelling a download targets its model without claiming completion', async () => {
  const view = await loaded([textModel('downloading')])
  await view.click('Cancel download')
  assert.equal(view.changes[0].operation, 'cancel')
  assert.equal(view.changes[0].modelId, 'lfm3-1.7b-bf16')
  view.changes[0].resolve([textModel('cancelling')])
  await view.flush()
  assert.equal(button(view.tree, 'Cancelling…').props.disabled, true)
  view.unmount()
})

test('uncertain model change requires a read, never automatic mutation replay', async () => {
  const view = await loaded()
  await view.click('Download')
  await acknowledge(view)
  await view.click('Accept and download', modal(view.tree))
  view.changes[0].reject(new Error('Timed out'))
  await view.flush()
  assert.match(alert(view.tree), /may have been saved/)
  assert.equal(
    button(modal(view.tree), 'Accept and download').props.disabled,
    true,
  )
  await view.click('Refresh models')
  view.reads.at(-1).resolve(saved([textModel('downloading')]))
  await view.flush()
  assert.equal(view.changes.length, 1)
  assert.equal(modal(view.tree), undefined)
  view.unmount()
})

test('context maximum is explicit and invalid limits never reach the API', async () => {
  const view = await loaded()
  await view.click('Context')
  const field = () =>
    elements(modal(view.tree)).find((node) => node.type === 'input')
  field().props.onChange({ target: { value: '0' } })
  await view.flush()
  await view.click('Save context', modal(view.tree))
  assert.equal(view.changes.length, 0)
  assert.match(alert(modal(view.tree)), /positive whole-token/)
  field().props.onChange({ target: { value: '' } })
  await view.flush()
  await view.click('Save context', modal(view.tree))
  assert.equal(view.changes[0].operation, 'context')
  assert.equal(view.changes[0].context, null)
  view.changes[0].resolve([{ ...textModel(), context_limit: null }])
  await view.flush()
  assert.equal(modal(view.tree), undefined)
  assert.match(content(view.tree), /Saved context:\s+model maximum/)
  view.unmount()
})

test('obsolete reads cannot replace a successful model mutation', async () => {
  const view = await loaded([textModel('downloading')])
  await view.poll()
  const old = view.reads.at(-1)
  await view.click('Cancel download')
  view.changes[0].resolve([textModel('cancelling')])
  await view.flush()
  old.resolve(saved())
  await view.flush()
  assert.equal(button(view.tree, 'Cancelling…').props.disabled, true)
  view.unmount()
})

test('unmount aborts pending model action and ignores its late response', async () => {
  const view = await loaded()
  await view.click('Download')
  await acknowledge(view)
  await view.click('Accept and download', modal(view.tree))
  const pending = view.changes[0]
  view.unmount()
  assert.equal(pending.signal.aborted, true)
  pending.resolve([textModel('complete')])
  await view.flush()
  assert.equal(view.updatesAfterUnmount, 0)
})

test('a saved null context opens as model maximum and can be replaced explicitly', async () => {
  const view = await loaded([{ ...textModel(), context_limit: null }])
  await view.click('Context')
  const input = elements(modal(view.tree)).find((node) => node.type === 'input')
  assert.equal(input.props.value, '')
  input.props.onChange({ target: { value: '16384' } })
  await view.flush()
  await view.click('Save context', modal(view.tree))
  assert.equal(view.changes[0].context, 16384)
  assert.equal(view.changes[0].modelId, 'lfm3-1.7b-bf16')
  view.changes[0].resolve([{ ...textModel(), context_limit: 16384 }])
  await view.flush()
  assert.match(content(view.tree), /Saved context:\s+16,384/)
  view.unmount()
})

test('failed context refresh keeps changes locked until authoritative recovery', async () => {
  const view = await loaded()
  await view.click('Context')
  elements(modal(view.tree))
    .find((node) => node.type === 'input')
    .props.onChange({ target: { value: '' } })
  await view.flush()
  await view.click('Save context', modal(view.tree))
  view.changes[0].reject(new Error('Save timed out'))
  await view.flush()
  assert.equal(button(modal(view.tree), 'Save context').props.disabled, true)
  await view.click('Refresh models')
  view.reads.at(-1).reject(new Error('Read unavailable'))
  await view.flush()
  assert.match(alert(view.tree), /Save timed out/)
  assert.match(alert(view.tree), /Read unavailable/)
  assert.equal(button(modal(view.tree), 'Save context').props.disabled, true)
  assert.equal(view.changes.length, 1)
  await view.click('Refresh models')
  view.reads.at(-1).resolve(saved([{ ...textModel(), context_limit: null }]))
  await view.flush()
  assert.equal(modal(view.tree), undefined)
  assert.match(content(view.tree), /Saved context:\s+model maximum/)
  assert.equal(view.changes.length, 1)
  view.unmount()
})

test('unmount aborts a pending context save and ignores its late failure', async () => {
  const view = await loaded()
  await view.click('Context')
  await view.click('Save context', modal(view.tree))
  const pending = view.changes[0]
  view.unmount()
  assert.equal(pending.signal.aborted, true)
  pending.reject(new Error('Late context failure'))
  await view.flush()
  assert.equal(view.updatesAfterUnmount, 0)
})
