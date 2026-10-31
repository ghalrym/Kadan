import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(new URL('../src/pages/TextToSpeechPage.tsx', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
})

function clip(script) {
  return { voice: 'Ryan', meta: 'WAV', script, time: '0:01', audio_base64: script, mime_type: 'audio/wav' }
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
const alert = tree => elements(tree).filter(node => node.props?.role === 'alert').map(content).join(' ')
const cards = tree => elements(tree).filter(node => node.type?.name === 'AudioCard').map(node => node.props)

// Invoke the page with local hook state, JSX records and deferred API mocks.
// Child components stay opaque; no DOM, media, model or service is invoked.
async function page() {
  const slots = [], effects = [], reads = [], requests = [], modelReads = []
  let cursor = 0, dirty = true, tree, mounted = true, updatesAfterUnmount = 0
  function request(calls, signal, body) {
    return new Promise((resolve, reject) => calls.push({ signal, body, resolve, reject }))
  }
  const context = createContext({ AbortController, Error })
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
  const modules = {
    react, 'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }),
    '../components/Media': module({ AudioCard: function AudioCard() {} }),
    '../data/playground': module({ speechScript: 'Editable script', voiceDescription: 'Calm' }),
    '../api/speech': module({
      fetchSpeechHistory: signal => request(reads, signal),
      requestSpeech: (body, signal) => request(requests, signal, body),
    }),
    '../api/generated/sdk.gen': module({ listSpeechModels: ({ signal }) => request(modelReads, signal) }),
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
  modelReads[0].resolve({ response: { ok: true }, data: [{
    id: 'qwen-tts-1.7b-custom', name: 'CustomVoice', speakers: ['Ryan'],
    default_speaker: 'Ryan', supports_instruction: true,
  }] })
  await flush()
  return {
    reads, requests, flush,
    get tree() { return tree },
    get updatesAfterUnmount() { return updatesAfterUnmount },
    async click(label) {
      const target = button(tree, label)
      assert.ok(target, `Missing button: ${label}`)
      assert.ok(!target.props.disabled, `Disabled button: ${label}`)
      target.props.onClick()
      await flush()
    },
    async submit() {
      elements(tree).find(node => node.type === 'form').props.onSubmit({ preventDefault() {} })
      await flush()
    },
    unmount() {
      mounted = false
      slots.forEach(slot => slot.cleanup?.())
      assert.equal(modelReads[0].signal.aborted, true)
    },
  }
}

async function failedHistory() {
  const view = await page()
  view.reads[0].reject(new Error('History unavailable'))
  await view.flush()
  assert.match(alert(view.tree), /History unavailable/)
  return view
}

async function confirm(view, script) {
  await view.submit()
  view.requests.at(-1).resolve(clip(script))
  await view.flush()
}

test('empty history retry preserves confirmed local audio and its newest-first order', async () => {
  const view = await failedHistory()
  await confirm(view, 'first')
  await confirm(view, 'second')
  const confirmed = cards(view.tree)
  assert.deepEqual(confirmed.map(item => item.script), ['second', 'first'])
  assert.deepEqual(confirmed.map(item => item.audioUrl), ['data:audio/wav;base64,second', 'data:audio/wav;base64,first'])
  await view.click('Retry history')
  assert.deepEqual(cards(view.tree), confirmed)
  view.reads[1].resolve([])
  await view.flush()
  assert.deepEqual(cards(view.tree), confirmed)
  assert.equal(alert(view.tree), '')
  assert.equal(button(view.tree, 'Retry history'), undefined)
  assert.equal(button(view.tree, 'Generate speech').props.disabled, false)
  assert.equal(view.requests.length, 2)
  view.unmount()
})

test('failed history retry retains local audio and recovers without resubmitting speech', async () => {
  const view = await failedHistory()
  await confirm(view, 'confirmed')
  const confirmed = cards(view.tree)
  await view.click('Retry history')
  view.reads[1].reject(new Error('History still unavailable'))
  await view.flush()
  assert.deepEqual(cards(view.tree), confirmed)
  assert.match(alert(view.tree), /History still unavailable/)
  assert.equal(button(view.tree, 'Generate speech').props.disabled, false)
  await view.click('Retry history')
  assert.equal(alert(view.tree), '')
  view.reads[2].resolve([])
  await view.flush()
  assert.deepEqual(cards(view.tree), confirmed)
  assert.equal(button(view.tree, 'Retry history'), undefined)
  assert.equal(view.requests.length, 1)
  view.unmount()
})

test('initial history keeps its returned order and new local clips precede it', async () => {
  const view = await page()
  assert.equal(cards(view.tree).length, 0)
  view.reads[0].resolve([clip('recent'), clip('older')])
  await view.flush()
  assert.deepEqual(cards(view.tree).map(item => item.script), ['recent', 'older'])
  await confirm(view, 'local')
  assert.deepEqual(cards(view.tree).map(item => item.script), ['local', 'recent', 'older'])
  assert.equal(button(view.tree, 'Retry history'), undefined)
  view.unmount()
})

test('empty initial history shows only the empty placeholder', async () => {
  const view = await page()
  view.reads[0].resolve([])
  await view.flush()
  assert.equal(cards(view.tree).length, 1)
  assert.equal(cards(view.tree)[0].voice, 'No generated audio')
  assert.equal(cards(view.tree)[0].audioUrl, undefined)
  view.unmount()
})

test('history loading blocks submission and pending speech disables history retry', async () => {
  const view = await page()
  assert.equal(button(view.tree, 'Generate speech').props.disabled, true)
  await view.submit()
  assert.equal(view.requests.length, 0)
  view.reads[0].reject(new Error('History unavailable'))
  await view.flush()
  await view.submit()
  assert.equal(button(view.tree, 'Retry history').props.disabled, true)
  await view.submit()
  assert.equal(view.requests.length, 1)
  view.requests[0].resolve(clip('confirmed'))
  await view.flush()
  await view.click('Retry history')
  assert.equal(view.reads[0].signal.aborted, true)
  assert.equal(button(view.tree, 'Generate speech').props.disabled, true)
  assert.equal(button(view.tree, 'Retry history'), undefined)
  await view.submit()
  assert.equal(view.requests.length, 1)
  view.reads[1].resolve([])
  await view.flush()
  assert.equal(button(view.tree, 'Generate speech').props.disabled, false)
  view.unmount()
})

test('cancelled speech cannot add late audio or release a newer pending request', async () => {
  const view = await failedHistory()
  await confirm(view, 'confirmed')
  await view.submit()
  const cancelled = view.requests.at(-1)
  await view.click('Cancel')
  assert.equal(cancelled.signal.aborted, true)
  await view.submit()
  cancelled.resolve(clip('cancelled'))
  await view.flush()
  assert.deepEqual(cards(view.tree).map(item => item.script), ['confirmed'])
  assert.ok(button(view.tree, 'Cancel'))
  assert.equal(button(view.tree, 'Retry history').props.disabled, true)
  view.requests.at(-1).resolve(clip('newer'))
  await view.flush()
  assert.deepEqual(cards(view.tree).map(item => item.script), ['newer', 'confirmed'])
  view.unmount()
})

for (const pending of ['initial history', 'history retry', 'speech']) {
  for (const settle of ['resolve', 'reject']) {
    test(`unmount aborts pending ${pending} and ignores its late ${settle}`, async () => {
      const view = pending === 'initial history' ? await page() : await failedHistory()
      if (pending === 'history retry') {
        await confirm(view, 'confirmed')
        await view.click('Retry history')
      }
      if (pending === 'speech') await view.submit()
      const request = pending === 'speech' ? view.requests.at(-1) : view.reads.at(-1)
      view.unmount()
      assert.equal(request.signal.aborted, true)
      request[settle](settle === 'reject' ? new Error('Late failure') : pending === 'speech' ? clip('late') : [clip('late')])
      await view.flush()
      assert.equal(view.updatesAfterUnmount, 0)
    })
  }
}
