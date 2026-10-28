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
const conversationModel = 'lfm3-1.7b-bf16'
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
    calls = []
  const context = createContext({
    AbortController,
    Error,
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
    calls,
    flush,
    get tree() {
      return tree
    },
    get lateUpdates() {
      return lateUpdates
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
test('Chat sends the released BF16 LFM3 model without waiting for Settings', async () => {
  const view = await page()
  assert.match(content(view.tree), /LFM3 1\.7B BF16/)
  await view.send('Hello')
  assert.equal(view.calls[0].args[4], conversationModel)
  view.unmount()
})
test('a missing LFM3 model fails visibly; retry and New chat keep LFM3', async () => {
  const view = await page()
  await view.send('Hello')
  const first = view.calls[0].args[3]
  view.calls[0].reject(
    new Error('Download LFM3 explicitly in Settings before chat.'),
  )
  await view.flush()
  assert.match(
    content(view.tree),
    /Download LFM3 explicitly in Settings before chat\./,
  )
  await view.click('Retry last message')
  assert.equal(view.calls[1].args[4], conversationModel)
  assert.equal(view.calls[1].args[3], first)
  assert.deepEqual(view.calls[1].args[0], view.calls[0].args[0])
  view.calls[1].resolve({ role: 'assistant', text: 'Hello' })
  await view.flush()
  await view.click('New chat')
  await view.send('A fresh conversation')
  assert.equal(view.calls[2].args[4], conversationModel)
  assert.notEqual(view.calls[2].args[3], first)
  view.unmount()
})
test('unmount aborts the chat request and ignores its late response', async () => {
  const view = await page()
  await view.send('Hello')
  view.unmount()
  assert.equal(view.calls[0].args[1].aborted, true)
  view.calls[0].resolve({ role: 'assistant', text: 'Late response' })
  await view.flush()
  assert.equal(view.lateUpdates, 0)
})
test('request builder preserves explicit LFM3 and omitted-model API requests', async () => {
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
  assert.equal(
    module.namespace.chatRequest(messages, conversationModel).model,
    conversationModel,
  )
})
