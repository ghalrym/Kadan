import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const source = await readFile(
  new URL('../src/pages/SpeechToTextPage.tsx', import.meta.url),
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
const button = (tree, label) =>
  elements(tree).find(
    (node) => node.type === 'button' && content(node) === label,
  )
const alerts = (tree) =>
  elements(tree)
    .filter((node) => node.props?.role === 'alert')
    .map(content)
    .join(' ')
const transcript = (tree) =>
  content(elements(tree).find((node) => node.props?.role === 'status'))
const copyError = 'Could not copy to the clipboard.'
const result = (text = 'Recognized text') => ({
  text,
  raw_text: text,
  formatting_status: 'not_requested',
})

// Invoke local hooks and JSX records; Panel, clipboard, conversion, requests,
// object URLs and timers are fakes. No renderer, DOM, audio or service is used.
async function page() {
  const slots = [],
    effects = [],
    writes = [],
    conversions = [],
    requests = [],
    timers = []
  let cursor = 0,
    dirty = true,
    tree,
    mounted = true,
    updatesAfterUnmount = 0,
    now = 0
  class File {
    constructor(name) {
      this.name = name
    }
  }
  function deferred(calls, data) {
    return new Promise((resolve, reject) =>
      calls.push({ ...data, resolve, reject }),
    )
  }
  const context = createContext({
    AbortController,
    Error,
    File,
    URL: {
      createObjectURL: (file) => `blob:fixture/${file.name}`,
      revokeObjectURL() {},
    },
    navigator: {
      clipboard: { writeText: (text) => deferred(writes, { text }) },
    },
    window: {
      setTimeout(callback, milliseconds) {
        const timer = {
          callback,
          milliseconds,
          due: now + milliseconds,
          active: true,
        }
        timers.push(timer)
        return timer
      },
      clearTimeout(timer) {
        timer.active = false
      },
    },
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
  const modules = {
    react,
    'react/jsx-runtime': module({ jsx, jsxs: jsx, Fragment: 'fragment' }),
    '../components/Controls': module({ Panel: function Panel() {} }),
    '../api/transcription': module({
      recordingToWav: (clip) => deferred(conversions, { clip }),
      requestTranscription: (audio, signal) =>
        deferred(requests, { audio, signal }),
    }),
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
        tree = component.namespace.default()
        effects.splice(0).forEach((run) => run())
      }
    }
    return tree
  }
  await flush()
  return {
    writes,
    conversions,
    requests,
    timers,
    flush,
    get tree() {
      return tree
    },
    get updatesAfterUnmount() {
      return updatesAfterUnmount
    },
    async click(label) {
      const target = button(tree, label)
      assert.ok(target, `Missing button: ${label}`)
      assert.ok(!target.props.disabled, `Disabled button: ${label}`)
      target.props.onClick()
      await flush()
    },
    async choose(name = 'inert fixture.wav') {
      const target = elements(tree).find(
        (node) => node.type === 'input' && node.props.type === 'file',
      )
      assert.ok(
        target && !target.props.disabled,
        'File choice must be present and enabled',
      )
      target.props.onChange({
        target: { files: [new File(name)], value: name },
      })
      await flush()
    },
    async recognize(value = result()) {
      conversions.at(-1).resolve('inert converted audio')
      await flush()
      requests.at(-1).resolve(value)
      await flush()
    },
    async advance(milliseconds) {
      now += milliseconds
      for (const timer of timers) {
        if (timer.active && timer.due <= now) {
          timer.active = false
          timer.callback()
        }
      }
      await flush()
    },
    unmount() {
      mounted = false
      slots.forEach((slot) => slot.cleanup?.())
    },
  }
}

async function recognized(value = result()) {
  const view = await page()
  await view.choose()
  await view.click('Transcribe')
  await view.recognize(value)
  return view
}

test('failed copy recovers on retry without leaving a clipboard error beside Copied', async () => {
  const view = await recognized()
  await view.click('Copy')
  view.writes[0].reject(new Error('Denied'))
  await view.flush()
  assert.equal(alerts(view.tree), copyError)
  await view.click('Copy')
  view.writes[1].resolve()
  await view.flush()
  assert.ok(button(view.tree, 'Copied'))
  assert.equal(alerts(view.tree), '')
  assert.equal(transcript(view.tree), 'Recognized text')
  view.unmount()
})

test('retry clears only clipboard feedback and preserves recognition error, warning and raw transcript', async () => {
  const view = await recognized({
    ...result('Raw text'),
    formatting_status: 'failed',
    formatting_error: 'Formatter unavailable.',
  })
  await view.click('Transcribe')
  view.conversions.at(-1).resolve('inert audio')
  await view.flush()
  view.requests.at(-1).reject(new Error('Recognition unavailable.'))
  await view.flush()
  const otherErrors = alerts(view.tree)
  assert.match(otherErrors, /Recognition unavailable/)
  assert.match(
    otherErrors,
    /Formatting failed:.*Formatter unavailable.*Raw transcription/,
  )
  await view.click('Copy')
  view.writes[0].reject(new Error('Denied'))
  await view.flush()
  assert.ok(alerts(view.tree).includes(copyError))
  assert.ok(alerts(view.tree).includes('Recognition unavailable.'))
  await view.click('Copy')
  assert.equal(alerts(view.tree), otherErrors)
  view.writes[1].resolve()
  await view.flush()
  assert.equal(alerts(view.tree), otherErrors)
  assert.equal(transcript(view.tree), 'Raw text')
  assert.ok(button(view.tree, 'Copied'))
  view.unmount()
})

test('copy writes the displayed formatted snapshot and confirms for 1500ms only after resolution', async () => {
  const view = await recognized({
    ...result('Formatted text'),
    raw_text: 'raw text',
    formatting_status: 'formatted',
  })
  await view.click('Copy')
  assert.deepEqual(
    view.writes.map((write) => write.text),
    ['Formatted text'],
  )
  assert.ok(button(view.tree, 'Copy'))
  assert.equal(view.timers.length, 0)
  view.writes[0].resolve()
  await view.flush()
  assert.ok(button(view.tree, 'Copied'))
  assert.equal(view.timers[0].milliseconds, 1500)
  await view.advance(1499)
  assert.ok(button(view.tree, 'Copied'))
  await view.advance(1)
  assert.ok(button(view.tree, 'Copy'))
  assert.equal(transcript(view.tree), 'Formatted text')
  assert.match(content(view.tree), /raw text/)
  view.unmount()
})

test('pending copy cannot restore feedback after Discard', async () => {
  const view = await recognized()
  await view.click('Copy')
  await view.click('Discard')
  assert.equal(transcript(view.tree), '')
  view.writes[0].reject(new Error('Late failure'))
  await view.flush()
  assert.equal(alerts(view.tree), '')
  assert.equal(button(view.tree, 'Copied'), undefined)
  await view.choose('replacement.wav')
  await view.click('Transcribe')
  await view.recognize()
  assert.ok(button(view.tree, 'Copy'))
  view.unmount()
})

test('file replacement clears success and fences a pending copy of the old transcript', async () => {
  const view = await recognized()
  await view.click('Copy')
  view.writes[0].resolve()
  await view.flush()
  await view.click('Copied')
  await view.choose('replacement.wav')
  await view.click('Transcribe')
  await view.recognize(result('Replacement'))
  view.writes[1].resolve()
  await view.flush()
  assert.ok(button(view.tree, 'Copy'))
  assert.equal(alerts(view.tree), '')
  assert.equal(transcript(view.tree), 'Replacement')
  view.unmount()
})

test('new transcription fences pending copy even when the new text is identical', async () => {
  const view = await recognized()
  await view.click('Copy')
  await view.click('Transcribe')
  view.writes[0].resolve()
  await view.flush()
  assert.ok(button(view.tree, 'Copy'))
  assert.equal(transcript(view.tree), 'Recognized text')
  await view.recognize()
  assert.ok(button(view.tree, 'Copy'))
  view.unmount()
})

test('accepting replacement text fences copy started while transcription was pending', async () => {
  const view = await recognized()
  await view.click('Transcribe')
  await view.click('Copy')
  await view.recognize()
  view.writes[0].reject(new Error('Late failure for identical old text'))
  await view.flush()
  assert.equal(alerts(view.tree), '')
  assert.ok(button(view.tree, 'Copy'))
  view.unmount()
})

test('new transcription and accepting a replacement each clear existing copy feedback', async () => {
  const view = await recognized()
  await view.click('Copy')
  view.writes[0].reject(new Error('Denied'))
  await view.flush()
  await view.click('Transcribe')
  assert.equal(alerts(view.tree), '')
  await view.click('Copy')
  view.writes[1].resolve()
  await view.flush()
  assert.ok(button(view.tree, 'Copied'))
  await view.recognize()
  assert.ok(button(view.tree, 'Copy'))
  view.unmount()
})

test('latest copy attempt owns feedback over older success and failure', async () => {
  const view = await recognized()
  await view.click('Copy')
  await view.click('Copy')
  view.writes[0].resolve()
  await view.flush()
  assert.ok(button(view.tree, 'Copy'))
  view.writes[1].reject(new Error('Latest failure'))
  await view.flush()
  assert.equal(alerts(view.tree), copyError)
  await view.click('Copy')
  assert.equal(alerts(view.tree), '')
  await view.click('Copy')
  view.writes[3].resolve()
  await view.flush()
  view.writes[2].reject(new Error('Older failure'))
  await view.flush()
  assert.ok(button(view.tree, 'Copied'))
  assert.equal(alerts(view.tree), '')
  view.unmount()
})

test('old expiry cannot clear a newer copy success or restore discarded feedback', async () => {
  const view = await recognized()
  await view.click('Copy')
  view.writes[0].resolve()
  await view.flush()
  const oldExpiry = view.timers[0].callback
  await view.advance(1000)
  await view.click('Copied')
  assert.ok(button(view.tree, 'Copy'))
  view.writes[1].resolve()
  await view.flush()
  oldExpiry()
  await view.flush()
  await view.advance(500)
  assert.ok(button(view.tree, 'Copied'))
  await view.advance(999)
  assert.ok(button(view.tree, 'Copied'))
  await view.advance(1)
  assert.ok(button(view.tree, 'Copy'))
  await view.click('Copy')
  view.writes[2].resolve()
  await view.flush()
  await view.click('Discard')
  await view.choose()
  await view.click('Transcribe')
  await view.recognize()
  assert.ok(button(view.tree, 'Copy'))
  view.unmount()
})

for (const settle of ['resolve', 'reject']) {
  test(`unmount ignores pending copy ${settle} and queued success expiry`, async () => {
    const view = await recognized()
    await view.click('Copy')
    view.writes[0].resolve()
    await view.flush()
    const expiry = view.timers[0].callback
    await view.click('Copied')
    view.unmount()
    view.writes[1][settle](
      settle === 'reject' ? new Error('Late failure') : undefined,
    )
    expiry()
    await view.flush()
    assert.equal(view.updatesAfterUnmount, 0)
  })
}

test('unmount ignores the current success expiry', async () => {
  const view = await recognized()
  await view.click('Copy')
  view.writes[0].resolve()
  await view.flush()
  const expiry = view.timers[0].callback
  view.unmount()
  expiry()
  await view.flush()
  assert.equal(view.updatesAfterUnmount, 0)
})
