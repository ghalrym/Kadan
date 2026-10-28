import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(new URL('../src/components/RequestDetails.tsx', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
})

function record(id = 'request-a') {
  return {
    id, type: 'Video', endpoint: '/v1/videos', model: 'fixture-model',
    status: 202, latency: '12 ms', latency_ms: 12, time: '10:00:00',
    prompt: 'Observed summary: "fixture"\nsecond line', output: 'Job accepted',
    request_bytes: 120, response_bytes: 80, video_job_id: 'fixture-job',
    video_status: 'Queued', output_tokens: null, cache_hit: false,
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
const button = (tree, label) => elements(tree).find(node => node.type === 'button' && content(node) === label)
const feedback = (tree, role) => elements(tree).filter(node => node.props?.role === role).map(content).join(' ')
const json = tree => elements(tree).find(node => node.type === 'pre')?.props.children

// Call the component with local hook state and JSX records; all external boundaries are fakes.
// No renderer, DOM, browser, real clipboard, fetch or service is used.
async function drawer({ request = record(), clipboard = 'deferred' } = {}) {
  const slots = [], effects = [], writes = [], polls = []
  let cursor = 0, dirty = true, tree, mounted = true, updatesAfterUnmount = 0, refreshes = 0
  let requestId = 'request-a'
  let polling = { data: request ? { request } : null, error: '' }
  const navigator = {}
  if (clipboard !== 'missing') navigator.clipboard = {
    writeText(text) {
      if (clipboard === 'throws') throw new Error('Clipboard unavailable')
      return new Promise((resolve, reject) => writes.push({ text, resolve, reject }))
    },
  }
  const context = createContext({ navigator, AbortController })
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
  function ActionButton() {}
  const modules = {
    react, 'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }),
    'react-router': module({ Link: 'Link', useParams: () => ({ requestId }) }),
    './Controls': module({ ActionButton }),
    '../api/requestStatus': module({ requestStatus: () => ({ label: 'Queued', className: 'queued' }) }),
    '../hooks/usePolling': module({ usePolling: url => {
      polls.push(url)
      return { ...polling, refresh: () => { refreshes++ } }
    } }),
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
    writes, polls, flush,
    get tree() { return tree },
    get refreshes() { return refreshes },
    get updatesAfterUnmount() { return updatesAfterUnmount },
    async click(label) {
      const target = button(tree, label)
      assert.ok(target, `Missing button: ${label}`)
      assert.ok(!target.props.disabled, `Disabled button: ${label}`)
      target.props.onClick()
      await flush()
    },
    async observe(request, error = '') {
      polling = { data: request ? { request } : null, error }
      dirty = true
      await flush()
    },
    async switchTo(id, request = null) {
      requestId = id
      await this.observe(request)
    },
    unmount() {
      mounted = false
      slots.forEach(slot => slot.cleanup?.())
    },
  }
}

test('copies exactly the Advanced JSON at click time and confirms only after clipboard resolution', async () => {
  const request = record()
  const view = await drawer({ request })
  const copy = button(view.tree, 'Copy details')
  assert.equal(copy.props.title, 'Copy observed request as JSON')
  assert.equal(copy.props.className, 'button button--secondary')
  assert.equal(copy.props.type, 'button')
  const snapshot = JSON.stringify(request, null, 2)
  assert.equal(json(view.tree), snapshot)
  assert.equal(view.writes.length, 0)
  await view.click('Copy details')
  assert.deepEqual(view.writes.map(write => write.text), [snapshot])
  assert.equal(feedback(view.tree, 'status'), '')
  assert.equal(feedback(view.tree, 'alert'), '')
  await view.observe({ ...request, video_status: 'Done', output: 'Finished' })
  assert.notEqual(json(view.tree), snapshot)
  assert.equal(view.writes[0].text, snapshot)
  view.writes[0].resolve()
  await view.flush()
  assert.equal(feedback(view.tree, 'status'), 'Details copied')
  await view.click('Copy details')
  assert.equal(view.writes[1].text, json(view.tree))
  assert.equal(feedback(view.tree, 'status'), '')
  assert.equal(view.refreshes, 0)
  assert.ok(view.polls.every(url => url === '/v1/requests/request-a'))
  assert.equal(button(view.tree, 'Copy as cURL'), undefined)
  const replay = elements(view.tree).find(node => node.type?.name === 'ActionButton')
  assert.equal(content(replay), 'Replay request')
  assert.equal(replay.props.onClick, undefined)
  view.unmount()
  view.writes[1].resolve()
  await view.flush()
  assert.equal(view.updatesAfterUnmount, 0)
})

for (const failure of ['rejects', 'throws', 'missing']) {
  test(`clipboard ${failure}: useful details failure leaves Advanced JSON available`, async () => {
    const view = await drawer({ clipboard: failure === 'rejects' ? 'deferred' : failure })
    const snapshot = json(view.tree)
    await view.click('Copy details')
    if (failure === 'rejects') {
      assert.equal(feedback(view.tree, 'alert'), '')
      view.writes[0].reject(new Error('Clipboard denied'))
      await view.flush()
    }
    assert.equal(feedback(view.tree, 'status'), '')
    assert.match(feedback(view.tree, 'alert'), /Could not copy details.*Select the JSON under Advanced.*manually/)
    assert.equal(json(view.tree), snapshot)
    assert.ok(elements(view.tree).some(node => node.type === 'details' && content(node).includes('Advanced')))
    await view.observe({ ...record(), video_status: 'Done' })
    assert.match(feedback(view.tree, 'alert'), /Could not copy details/)
    view.unmount()
  })
}

test('a failed copy can be retried and success replaces its failure only after resolution', async () => {
  const view = await drawer()
  await view.click('Copy details')
  view.writes[0].reject(new Error('Denied'))
  await view.flush()
  assert.match(feedback(view.tree, 'alert'), /Could not copy details/)
  await view.click('Copy details')
  assert.equal(feedback(view.tree, 'alert'), '')
  assert.equal(feedback(view.tree, 'status'), '')
  view.writes[1].resolve()
  await view.flush()
  assert.equal(feedback(view.tree, 'status'), 'Details copied')
  view.unmount()
})

for (const error of ['', 'Record not found. It may have expired or the API restarted.']) {
  test(`an unavailable record offers no copy or replay actions (${error ? 'failure' : 'loading'})`, async () => {
    const view = await drawer({ request: null })
    await view.observe(null, error)
    assert.equal(button(view.tree, 'Copy details'), undefined)
    assert.equal(button(view.tree, 'Copy ID'), undefined)
    assert.equal(json(view.tree), undefined)
    assert.equal(elements(view.tree).some(node => node.type?.name === 'ActionButton'), false)
    assert.equal(view.writes.length, 0)
    if (error) {
      assert.equal(feedback(view.tree, 'alert'), error)
      await view.click('Retry')
      assert.equal(view.refreshes, 1)
    } else assert.match(feedback(view.tree, 'status'), /Loading request/)
    view.unmount()
  })
}

for (const label of ['Copy details', 'Copy ID']) {
  for (const settle of ['resolve', 'reject']) {
    test(`${label} ignores late ${settle} after route or observed record identity changes`, async () => {
      for (const change of ['route', 'record']) {
        const view = await drawer()
        await view.click(label)
        const old = view.writes[0]
        if (change === 'route') await view.switchTo('request-b', record())
        else await view.observe(record('request-b'))
        assert.equal(feedback(view.tree, 'status'), '')
        assert.equal(feedback(view.tree, 'alert'), '')
        old[settle](settle === 'reject' ? new Error('Old failure') : undefined)
        await view.flush()
        assert.equal(feedback(view.tree, 'status'), '')
        assert.equal(feedback(view.tree, 'alert'), '')
        await view.switchTo('request-a', record())
        assert.equal(feedback(view.tree, 'status'), '')
        assert.equal(feedback(view.tree, 'alert'), '')
        view.unmount()
      }
    })

    test(`${label} ignores late ${settle} after unmount`, async () => {
      const view = await drawer()
      await view.click(label)
      view.unmount()
      view.writes[0][settle](settle === 'reject' ? new Error('Late failure') : undefined)
      await view.flush()
      assert.equal(view.updatesAfterUnmount, 0)
    })
  }
}

for (const first of ['Copy details', 'Copy ID']) {
  for (const second of ['Copy details', 'Copy ID']) {
    test(`${first} cannot confirm while a newer ${second} is still pending`, async () => {
      const view = await drawer()
      await view.click(first)
      await view.click(second)
      view.writes[0].resolve()
      await view.flush()
      assert.equal(feedback(view.tree, 'status'), '')
      assert.equal(feedback(view.tree, 'alert'), '')
      view.writes[1].resolve()
      await view.flush()
      assert.equal(feedback(view.tree, 'status'), second === 'Copy details' ? 'Details copied' : 'ID copied')
      view.unmount()
    })

    for (const settle of ['resolve', 'reject']) {
      test(`${second} owns feedback over earlier ${first} ${settle}`, async () => {
        const view = await drawer()
        await view.click(first)
        await view.click(second)
        // The latest failure must survive an older success as well as an older failure.
        view.writes[1].reject(new Error('Latest failure'))
        await view.flush()
        const message = feedback(view.tree, 'alert')
        assert.match(message, second === 'Copy details' ? /Could not copy details/ : /Could not copy request ID/)
        view.writes[0][settle](settle === 'reject' ? new Error('Old failure') : undefined)
        await view.flush()
        assert.equal(feedback(view.tree, 'alert'), message)
        assert.equal(feedback(view.tree, 'status'), '')
        view.unmount()
      })
    }
  }
}

for (const label of ['Copy details', 'Copy ID']) {
  test(`${label} feedback does not return after switching away and back`, async () => {
    const view = await drawer()
    await view.click(label)
    view.writes[0].resolve()
    await view.flush()
    assert.match(feedback(view.tree, 'status'), /copied/)
    await view.switchTo('request-b', record('request-b'))
    assert.equal(feedback(view.tree, 'status'), '')
    await view.switchTo('request-a', record())
    assert.equal(feedback(view.tree, 'status'), '')
    assert.equal(feedback(view.tree, 'alert'), '')
    view.unmount()
  })
}

test('Copy ID writes only the displayed identifier and confirms after resolution', async () => {
  const view = await drawer()
  await view.click('Copy ID')
  assert.deepEqual(view.writes.map(write => write.text), ['request-a'])
  assert.equal(feedback(view.tree, 'status'), '')
  view.writes[0].resolve()
  await view.flush()
  assert.equal(feedback(view.tree, 'status'), 'ID copied')
  assert.equal(feedback(view.tree, 'alert'), '')
  view.unmount()
})

test('polling failure stays separate from a pending copy and recovery does not revive copy feedback', async () => {
  const view = await drawer()
  await view.click('Copy details')
  await view.observe(null, 'Monitoring API returned HTTP 503.')
  view.writes[0].reject(new Error('Clipboard denied'))
  await view.flush()
  assert.equal(feedback(view.tree, 'alert'), 'Monitoring API returned HTTP 503.')
  assert.ok(button(view.tree, 'Retry'))
  await view.observe(record())
  assert.equal(feedback(view.tree, 'alert'), '')
  assert.equal(feedback(view.tree, 'status'), '')
  assert.equal(json(view.tree), JSON.stringify(record(), null, 2))
  view.unmount()
})
