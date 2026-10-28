import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(new URL('../src/pages/ApiAccessPage.tsx', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
})

function schema() {
  return {
    info: { title: 'Hikari', version: '1.0', description: 'Available operations' },
    paths: { '/example': {
      get: { summary: 'Read example', description: 'Summary takes precedence' },
      post: { description: 'Create example' },
      put: {}, patch: { summary: 'Edit example' }, delete: { summary: 'Delete example' },
      parameters: [], head: { summary: 'Not listed' },
    } },
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
const button = (tree, label) => elements(tree).find(node => node.type === 'button' && content(node) === label)
const table = tree => elements(tree).find(node => node.type === 'table')
const status = tree => elements(tree).find(node => node.props?.role === 'status')

// Call the page with local hook state, JSX records and deferred fetch mocks.
// No React renderer, DOM, browser or service is used.
async function page() {
  const slots = [], effects = [], requests = [], timeouts = []
  let cursor = 0, dirty = true, tree, mounted = true, updatesAfterUnmount = 0
  const context = createContext({
    AbortController, Error,
    AbortSignal: {
      any: signals => AbortSignal.any(signals),
      timeout: milliseconds => {
        const controller = new AbortController()
        timeouts.push({ milliseconds, controller })
        return controller.signal
      },
    },
    fetch: (url, { signal }) => new Promise((resolve, reject) => requests.push({ url, signal, resolve, reject })),
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
    useEffect(callback, deps) {
      const index = cursor++, previous = slots[index]
      if (!previous || deps.some((value, i) => !Object.is(value, previous.deps[i]))) {
        slots[index] = { deps, cleanup: previous?.cleanup }
        effects.push(() => { slots[index].cleanup?.(); slots[index].cleanup = callback() })
      }
    },
  })
  const jsx = (type, props) => ({ type, props })
  const modules = { react, 'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }) }
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
    requests, timeouts, flush,
    get tree() { return tree },
    get updatesAfterUnmount() { return updatesAfterUnmount },
    async respond(data) {
      requests.at(-1).resolve({ ok: true, json: async () => data })
      await flush()
    },
    async retry() {
      const target = button(tree, 'Retry')
      assert.ok(target, 'Missing Retry button')
      target.props.onClick()
      await flush()
    },
    unmount() {
      mounted = false
      slots.forEach(slot => slot.cleanup?.())
    },
  }
}

for (const [label, change, message] of [
  ['object title', data => { data.info.title = {} }, /invalid OpenAPI schema/],
  ['empty title', data => { data.info.title = '' }, /invalid OpenAPI schema/],
  ['numeric title', data => { data.info.title = 1 }, /invalid OpenAPI schema/],
  ['object version', data => { data.info.version = {} }, /invalid OpenAPI schema/],
  ['missing version', data => { delete data.info.version }, /invalid OpenAPI schema/],
  ['object info description', data => { data.info.description = {} }, /invalid OpenAPI schema/],
  ['null info description', data => { data.info.description = null }, /invalid OpenAPI schema/],
  ['array paths', data => { data.paths = [] }, /invalid OpenAPI schema/],
  ['array path record', data => { data.paths['/example'] = [] }, /invalid path definitions/],
  ['null path record', data => { data.paths['/example'] = null }, /invalid path definitions/],
  ['array operation', data => { data.paths['/example'].get = [] }, /invalid operation definition/],
  ['null operation', data => { data.paths['/example'].get = null }, /invalid operation definition/],
  ['object summary', data => { data.paths['/example'].get.summary = {} }, /invalid operation definition/],
  ['object operation description', data => { data.paths['/example'].get.description = {} }, /invalid operation definition/],
]) {
  test(`rejects ${label} with a visible failure and Retry before publishing the schema`, async () => {
    const view = await page()
    const data = schema()
    change(data)
    await view.respond(data)
    assert.match(alert(view.tree), message)
    assert.ok(button(view.tree, 'Retry'))
    assert.equal(table(view.tree), undefined)
    assert.equal(status(view.tree), undefined)
    view.unmount()
  })
}

test('lists valid schema metadata and supported operations with explicit reference links', async () => {
  const view = await page()
  assert.match(content(status(view.tree)), /Loading API schema/)
  assert.equal(view.requests.length, 1)
  assert.equal(view.requests[0].url, '/openapi.json')
  await view.respond(schema())
  assert.match(content(view.tree), /Hikari\s+·\s+1.0/)
  assert.match(content(view.tree), /Available operations/)
  const rows = elements(table(view.tree)).filter(node => node.type === 'tr').slice(1)
  assert.deepEqual(rows.map(row => elements(row).filter(node => node.type === 'td').map(content)), [
    ['GET', '/example', 'Read example'], ['POST', '/example', 'Create example'],
    ['PUT', '/example', ''], ['PATCH', '/example', 'Edit example'], ['DELETE', '/example', 'Delete example'],
  ])
  assert.deepEqual(elements(view.tree).filter(node => node.type === 'a')
    .map(node => [node.props.href, node.props.target, node.props.rel, content(node)]), [
    ['/docs', '_blank', 'noreferrer', 'Interactive API reference'],
    ['/openapi.json', '_blank', 'noreferrer', 'OpenAPI JSON'],
  ])
  assert.equal(alert(view.tree), '')
  assert.equal(button(view.tree, 'Retry'), undefined)
  assert.equal(status(view.tree), undefined)
  view.unmount()
})

test('accepts an omitted description, empty string version and empty paths record', async () => {
  const view = await page()
  await view.respond({ info: { title: 'Hikari', version: '' }, paths: {} })
  assert.ok(table(view.tree))
  assert.equal(alert(view.tree), '')
  view.unmount()
})

for (const failure of ['transport', 'HTTP', 'JSON', 'timeout']) {
  test(`${failure} failure stays visible and retry aborts the old request before recovery`, async () => {
    const view = await page()
    const request = view.requests[0]
    if (failure === 'transport') request.reject(new Error('Server unreachable'))
    if (failure === 'HTTP') request.resolve({ ok: false, status: 503 })
    if (failure === 'JSON') request.resolve({ ok: true, json: async () => { throw new Error('Invalid JSON') } })
    if (failure === 'timeout') {
      assert.equal(view.timeouts[0].milliseconds, 15000)
      view.timeouts[0].controller.abort(new Error('Schema timed out'))
      assert.equal(request.signal.aborted, true)
      request.reject(request.signal.reason)
    }
    await view.flush()
    assert.match(alert(view.tree), {
      transport: /Server unreachable/, HTTP: /HTTP 503/, JSON: /Invalid JSON/, timeout: /Schema timed out/,
    }[failure])
    assert.equal(table(view.tree), undefined)
    await view.retry()
    assert.equal(request.signal.aborted, true)
    assert.equal(view.requests.length, 2)
    assert.equal(view.requests[1].signal.aborted, false)
    assert.equal(alert(view.tree), '')
    assert.match(content(status(view.tree)), /Loading API schema/)
    await view.respond(schema())
    assert.ok(table(view.tree))
    assert.equal(alert(view.tree), '')
    view.unmount()
  })
}

test('invalid schema can fail again on retry and then recover with a valid list', async () => {
  const view = await page()
  await view.respond({ ...schema(), paths: [] })
  assert.ok(button(view.tree, 'Retry'))
  await view.retry()
  await view.respond({ ...schema(), info: { title: {} } })
  assert.match(alert(view.tree), /invalid OpenAPI schema/)
  assert.equal(table(view.tree), undefined)
  await view.retry()
  await view.respond(schema())
  assert.ok(table(view.tree))
  assert.equal(button(view.tree, 'Retry'), undefined)
  view.unmount()
})

for (const pending of ['fetch', 'JSON', 'retry']) {
  for (const settle of ['resolve', 'reject']) {
    test(`unmount aborts pending ${pending} and ignores its late ${settle}`, async () => {
      const view = await page()
      if (pending === 'retry') {
        view.requests[0].reject(new Error('Server unreachable'))
        await view.flush()
        await view.retry()
      }
      const request = view.requests.at(-1)
      let completion = request
      if (pending === 'JSON') {
        const body = new Promise((resolve, reject) => { completion = { resolve, reject } })
        request.resolve({ ok: true, json: () => body })
        await view.flush()
      }
      view.unmount()
      assert.equal(request.signal.aborted, true)
      completion[settle](settle === 'reject' ? new Error('Late failure')
        : pending === 'JSON' ? schema() : { ok: true, json: async () => schema() })
      await view.flush()
      assert.equal(view.updatesAfterUnmount, 0)
    })
  }
}
