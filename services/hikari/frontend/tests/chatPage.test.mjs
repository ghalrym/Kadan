import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(
  new URL('../src/pages/ChatPage.tsx', import.meta.url),
  'utf8',
)
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
})
const ready = {
  id: 'lfm3-1.7b-bf16',
  display_name: 'LFM3 trial',
  kind: 'llm',
  status: 'complete',
  inference_available: true,
}
function elements(node) {
  return !node || typeof node !== 'object'
    ? []
    : [node, ...[node.props?.children].flat(Infinity).flatMap(elements)]
}
function content(node) {
  if (Array.isArray(node)) return node.map(content).join('')
  return node && typeof node === 'object'
    ? content(node.props?.children)
    : node == null || typeof node === 'boolean'
      ? ''
      : String(node)
}
function deferred() {
  let resolve, reject
  const promise = new Promise((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
async function page() {
  let cursor = 0,
    dirty = true,
    tree,
    mounted = true,
    serial = 0,
    lateUpdates = 0
  const slots = [],
    effects = [],
    calls = [],
    read = deferred()
  const context = createContext({
    AbortController,
    Uint8Array,
    crypto: { getRandomValues: (bytes) => bytes.fill(++serial) },
  })
  const module = (exports) =>
    new SyntheticModule(
      Object.keys(exports),
      function () {
        for (const [name, value] of Object.entries(exports))
          this.setExport(name, value)
      },
      { context },
    )
  const jsx = (type, props) => ({ type, props })
  const modules = {
    react: module({
      useState(initial) {
        const index = cursor++
        slots[index] ??= { value: initial }
        return [
          slots[index].value,
          (next) => {
            if (!mounted) lateUpdates++
            slots[index].value =
              typeof next === 'function' ? next(slots[index].value) : next
            dirty = true
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
    }),
    'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }),
    '../api/settings': module({
      fetchExperimentalModels: (signal) => {
        read.signal = signal
        return read.promise
      },
    }),
    '../api/chat': module({
      streamChat: (...args) => {
        const call = { args, ...deferred() }
        calls.push(call)
        return call.promise
      },
    }),
  }
  const component = new SourceTextModule(outputText, { context })
  await component.link((name) => {
    assert.ok(modules[name], name)
    return modules[name]
  })
  await component.evaluate()
  async function flush() {
    for (let i = 0; i < 12; i++) {
      await Promise.resolve()
      if (dirty && mounted) {
        dirty = false
        cursor = 0
        tree = component.namespace.default()
        effects.splice(0).forEach((run) => run())
      }
    }
  }
  await flush()
  return {
    read,
    calls,
    flush,
    get tree() {
      return tree
    },
    get lateUpdates() {
      return lateUpdates
    },
    get select() {
      return elements(tree).find((node) => node.type === 'select')
    },
    async choose(value) {
      this.select.props.onChange({ target: { value } })
      await flush()
    },
    async send(text) {
      elements(tree)
        .find((node) => node.type === 'textarea')
        .props.onChange({ target: { value: text } })
      await flush()
      elements(tree)
        .find((node) => node.type === 'form')
        .props.onSubmit({ preventDefault() {} })
      await flush()
    },
    async click(label) {
      const button = elements(tree).find(
        (node) => node.type === 'button' && content(node) === label,
      )
      assert.ok(button, label)
      assert.ok(!button.props.disabled, label)
      button.props.onClick()
      await flush()
    },
    unmount() {
      mounted = false
      slots.forEach((slot) => slot.cleanup?.())
    },
  }
}
test('offers only downloaded inference-ready LFM3 text; keeps Settings as default', async () => {
  const view = await page()
  view.read.resolve([
    ready,
    { ...ready, id: 'lfm3-audio-2b-bf16' },
    { ...ready, status: 'not_downloaded' },
    { ...ready, inference_available: false },
  ])
  await view.flush()
  assert.equal(view.select.props.value, '')
  assert.deepEqual(
    elements(view.select)
      .filter((node) => node.type === 'option')
      .map((node) => node.props.value),
    ['', ready.id],
  )
  view.unmount()
})
test('retry keeps model; New chat unlocks selection and resets conversation identity', async () => {
  const view = await page()
  view.read.resolve([ready])
  await view.flush()
  await view.choose(ready.id)
  await view.send('Hello')
  assert.equal(view.calls[0].args[4], ready.id)
  const first = view.calls[0].args[3]
  assert.equal(view.select.props.disabled, true)
  view.calls[0].reject(new Error('Unavailable'))
  await view.flush()
  assert.equal(view.select.props.disabled, true)
  await view.click('Retry last message')
  assert.equal(view.calls[1].args[4], ready.id)
  assert.equal(view.calls[1].args[3], first)
  view.calls[1].resolve({ role: 'assistant', text: 'Hello' })
  await view.flush()
  await view.click('New chat')
  assert.equal(view.select.props.disabled, false)
  await view.choose('')
  await view.send('Native')
  assert.equal(view.calls[2].args[4], undefined)
  assert.notEqual(view.calls[2].args[3], first)
  view.unmount()
})
test('unmount aborts model read and ignores late results', async () => {
  const view = await page()
  view.unmount()
  assert.equal(view.read.signal.aborted, true)
  view.read.resolve([ready])
  await view.flush()
  assert.equal(view.lateUpdates, 0)
})
test('request builder sends explicit model only when selected', async () => {
  const source = await readFile(
    new URL('../src/api/chat.ts', import.meta.url),
    'utf8',
  )
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link(() => {
    throw new Error('Unexpected runtime dependency')
  })
  await module.evaluate()
  const messages = [{ role: 'user', text: 'Hello' }]
  assert.equal('model' in module.namespace.chatRequest(messages), false)
  assert.equal(module.namespace.chatRequest(messages, ready.id).model, ready.id)
})
