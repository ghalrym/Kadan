import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'

async function moduleAt(path, dependencies = {}) {
  const source = await readFile(new URL(path, import.meta.url), 'utf8')
  const module = new SourceTextModule(stripTypeScriptTypes(source), { initializeImportMeta: meta => { meta.url = new URL(path, import.meta.url).href } })
  await module.link(specifier => {
    const exports = dependencies[specifier]
    assert.ok(exports, `Unexpected import: ${specifier}`)
    return new SyntheticModule(Object.keys(exports), function () { for (const [key, value] of Object.entries(exports)) this.setExport(key, value) })
  })
  await module.evaluate()
  return module.namespace
}
const audio = await moduleAt('../src/audio/voiceAudio.ts')
const api = await moduleAt('../src/api/voice.ts')
const history = await moduleAt('../src/audio/voiceHistory.ts')

test('VAD requires sustained activity and preserves pre-roll through the endpoint', () => {
  const vad = new audio.VoiceActivity(1000)
  const silence = new Float32Array(20)
  for (let i = 0; i < 30; i++) assert.equal(vad.push(silence).started, false)
  const speech = new Float32Array(20).fill(0.1)
  assert.equal(vad.push(speech).started, false)
  assert.equal(vad.push(speech).started, false)
  assert.equal(vad.push(speech).started, true)
  assert.ok(vad.snapshot().length < 400)
  let result
  for (let i = 0; i < 30; i++) result = vad.push(silence)
  assert.ok(result.ended.length > 600)
  assert.equal(vad.snapshot(), null)
})

test('VAD bounds uninterrupted speech to twenty seconds', () => {
  const vad = new audio.VoiceActivity(1000)
  let ended
  for (let i = 0; i < 1000 && !ended; i++) ended = vad.push(new Float32Array(20).fill(0.1)).ended
  assert.ok(ended)
  assert.ok(ended.length <= 20000)
})

test('WAV encoding preserves rate, mono samples, signed clipping and byte lengths', () => {
  const url = audio.wavDataUrl(new Float32Array([-2, 0, 2]), 16000)
  const bytes = Buffer.from(url.split(',')[1], 'base64')
  assert.equal(bytes.toString('ascii', 0, 4), 'RIFF')
  assert.equal(bytes.readUInt32LE(24), 16000)
  assert.equal(bytes.readUInt16LE(22), 1)
  assert.equal(bytes.readUInt32LE(40), 6)
  assert.deepEqual([44, 46, 48].map(offset => bytes.readInt16LE(offset)), [-32768, 0, 32767])
})

test('recognition windows preserve duration and stay below the request bound at common device rates', () => {
  for (const rate of [44100, 48000, 96000]) {
    const encoded = audio.recognitionWav(new Float32Array(rate * 20).fill(0.25), rate)
    assert.ok(encoded.length < 2 * 1024 * 1024)
    const bytes = Buffer.from(encoded.split(',')[1], 'base64')
    assert.equal(bytes.readUInt32LE(24), 16000)
    assert.equal(bytes.readUInt32LE(40), 16000 * 20 * 2)
    assert.equal(bytes.readInt16LE(44), 8192)
  }
})

function fakeContext() {
  const sources = []
  return { currentTime: 0, baseLatency: 0.01, outputLatency: 0.02, destination: {}, sources,
    createBuffer: (_channels, length, sampleRate) => ({ sampleRate, getChannelData: () => new Float32Array(length) }),
    createBufferSource: () => {
      const source = { connect() {}, disconnect() { this.disconnected = true }, start(at) { this.at = at }, stop() { this.stopped = true } }
      sources.push(source); return source
    },
  }
}
const pcm = seconds => Buffer.alloc(seconds * 8000 * 2).toString('base64')

test('PCM schedules continuous chunks, detects underruns and stops all sources', async () => {
  const context = fakeContext()
  const player = new audio.VoicePlayer(context)
  const signal = new AbortController().signal
  await player.push(pcm(1), 8000, signal)
  await player.push(pcm(1), 8000, signal)
  assert.equal(context.sources[0].at, 0.04)
  assert.equal(context.sources[1].at, 1.04)
  context.currentTime = 2.2
  await player.push(pcm(1), 8000, signal)
  assert.equal(player.underruns, 1)
  assert.ok(player.gapMs > 150)
  player.stop()
  assert.equal(player.playing, false)
  assert.ok(context.sources.every(source => source.stopped && source.disconnected))
})

test('PCM backpressure can be cancelled without scheduling stale audio', async () => {
  const context = fakeContext()
  const player = new audio.VoicePlayer(context)
  const controller = new AbortController()
  await player.push(pcm(4), 8000, controller.signal)
  const blocked = player.push(pcm(2), 8000, controller.signal)
  controller.abort()
  player.stop()
  await assert.rejects(blocked, { name: 'AbortError' })
  assert.equal(context.sources.length, 1)
})

test('PCM rejects malformed and oversized chunks', async () => {
  const player = new audio.VoicePlayer(fakeContext())
  for (const value of ['', btoa('x'), pcm(6)]) await assert.rejects(player.push(value, 8000, new AbortController().signal), /Invalid PCM/)
})

test('SSE preserves fragmented CRLF and waits for the audio consumer', async () => {
  const originalFetch = globalThis.fetch
  let cancelled = false
  globalThis.fetch = async () => new Response(new ReadableStream({
    start(controller) {
      for (const chunk of ['data: {"type":"text","text":"Hello"}\r', '\n\r\ndata: {"type":"audio","audio":"AAA=","sample_rate":8000}\n\n', 'data: {"type":"done","timing":{"tts_ms":12}}\n\n']) controller.enqueue(new TextEncoder().encode(chunk))
    }, cancel() { cancelled = true },
  }), { headers: { 'Content-Type': 'text/event-stream' } })
  try {
    const events = []
    await api.streamVoice('go', 'current', [{ role: 'user', content: 'Hi' }], new AbortController().signal, async event => { await Promise.resolve(); events.push(event.type) })
    assert.deepEqual(events, ['text', 'audio', 'done'])
    assert.equal(cancelled, true)
  } finally { globalThis.fetch = originalFetch }
})

test('SSE rejects premature EOF and backend errors instead of accepting partial success', async () => {
  const originalFetch = globalThis.fetch
  try {
    for (const [content, expected] of [
      ['data: {"type":"text","text":"partial"}\n\n', /before completion/],
      ['data: {"type":"error","detail":"Checkpoint unavailable"}\n\n', /Checkpoint unavailable/],
    ]) {
      globalThis.fetch = async () => new Response(content, { headers: { 'Content-Type': 'text/event-stream' } })
      await assert.rejects(api.streamVoice('go', 'drama3', [], new AbortController().signal, () => {}), expected)
    }
  } finally { globalThis.fetch = originalFetch }
})

const capabilities = { tier: 'go', asr_model: 'whisper', available: true, backends: [{ id: 'current', speech_model: 'qwen', available: true }] }

test('interruption suppresses a late transcript even if its transport ignores abort', async () => {
  let finishTranscript
  let chatCalls = 0
  const { VoiceSession } = await moduleAt('../src/audio/voiceSession.ts', {
    '../api/voice': { transcribeVoice: () => new Promise(resolve => { finishTranscript = resolve }), streamVoice: () => { chatCalls++ } },
    './voiceAudio': audio,
    './voiceHistory': history,
  })
  const snapshots = []
  const session = new VoiceSession(capabilities, 'current', 0.02, state => snapshots.push(state))
  session.context = { sampleRate: 16000 }
  const turn = session.respond(new Float32Array(10), performance.now())
  session.interrupt()
  finishTranscript('Stale text')
  await turn
  assert.equal(chatCalls, 0)
  assert.equal(snapshots.at(-1).measurements[0].status, 'interrupted')
})

test('stop while microphone permission is pending releases late granted tracks', async () => {
  const original = { window: globalThis.window, navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator'), AudioContext: globalThis.AudioContext, AudioWorkletNode: globalThis.AudioWorkletNode }
  let grant
  let closed = false
  let trackStopped = false
  globalThis.window = { isSecureContext: true }
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { mediaDevices: { getUserMedia: () => new Promise(resolve => { grant = resolve }) } } })
  globalThis.AudioWorkletNode = class {}
  globalThis.AudioContext = class { async resume() {} async close() { closed = true } }
  try {
    const { VoiceSession } = await moduleAt('../src/audio/voiceSession.ts', { '../api/voice': api, './voiceAudio': audio, './voiceHistory': history })
    const session = new VoiceSession(capabilities, 'current', 0.02, () => {})
    const started = session.start()
    await Promise.resolve()
    session.stop()
    grant({ getTracks: () => [{ stop() { trackStopped = true } }] })
    await started
    assert.equal(closed, true)
    assert.equal(trackStopped, true)
  } finally {
    globalThis.window = original.window
    Object.defineProperty(globalThis, 'navigator', original.navigator)
    globalThis.AudioContext = original.AudioContext; globalThis.AudioWorkletNode = original.AudioWorkletNode
  }
})

test('history prunes complete pairs and reserves space for the current user', () => {
  const context = new history.VoiceHistory()
  for (let i = 0; i < 25; i++) context.commit(`question ${i}`, `answer ${i}`, false)
  const messages = context.request('next question')
  assert.equal(messages.length, 39)
  assert.deepEqual(messages[0], { role: 'user', content: 'question 6' })
  assert.deepEqual(messages[1], { role: 'assistant', content: 'answer 6' })
  assert.deepEqual(messages.at(-1), { role: 'user', content: 'next question' })
  assert.ok(messages.every((message, index) => message.role === (index % 2 ? 'assistant' : 'user')))
})

test('history enforces UTF-8 bytes rather than string length and recovers after an oversized utterance', () => {
  const context = new history.VoiceHistory()
  for (let i = 0; i < 3; i++) context.commit('🙂'.repeat(2000), '界'.repeat(2000) + i, false)
  const messages = context.request('é'.repeat(3000))
  assert.equal(messages.length, 3)
  assert.ok(messages.reduce((sum, message) => sum + Buffer.byteLength(message.content), 0) <= 32000)
  assert.ok(messages[1].content.endsWith('2'))
  assert.throws(() => context.request('a'.repeat(8001)), /shorter utterance/)
  assert.deepEqual(context.request('retry').slice(0, -1), messages.slice(0, -1))
})

test('context truncation preserves the interruption annotation and the per-message character bound', () => {
  const context = new history.VoiceHistory()
  context.commit('question', 'answer '.repeat(2000), true)
  const response = context.request('next')[1].content
  assert.ok(response.startsWith('[Speech playback stopped before completion.'))
  assert.ok(response.endsWith('[Response truncated in conversation context.]'))
  assert.equal(Array.from(response).length, 8000)
})

test('barge-in during playback commits one annotated response pair for the next request', async () => {
  const calls = []
  let drainReached
  const draining = new Promise(resolve => { drainReached = resolve })
  let transcriptions = 0
  const { VoiceSession } = await moduleAt('../src/audio/voiceSession.ts', {
    '../api/voice': {
      transcribeVoice: async () => ++transcriptions === 1 ? 'First question' : 'Follow-up',
      streamVoice: async (_tier, _backend, messages, _signal, onEvent) => {
        calls.push(messages)
        await onEvent({ type: 'text', text: 'Available response text' })
        await onEvent({ type: 'audio', audio: 'AAA=', sample_rate: 8000 })
        await onEvent({ type: 'done' })
      },
    },
    './voiceAudio': audio, './voiceHistory': history,
  })
  const snapshots = []
  const session = new VoiceSession(capabilities, 'current', 0.02, state => snapshots.push(state))
  session.context = { sampleRate: 16000 }
  session.player = {
    seconds: 0, underruns: 0, gapMs: 0, playbackBegan: false,
    async push() { this.seconds = 1; this.firstScheduledAt = performance.now(); this.playbackBegan = true },
    drain(signal) {
      if (calls.length > 1) return Promise.resolve()
      drainReached()
      return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }))
    },
    stop() { this.seconds = 0; this.playbackBegan = false },
  }
  const first = session.respond(new Float32Array(10), performance.now())
  await draining
  session.interrupt()
  await first
  assert.equal(snapshots.at(-1).measurements[0].status, 'interrupted')
  await session.respond(new Float32Array(10), performance.now())
  assert.equal(calls[1].length, 3)
  assert.equal(calls[1][0].content, 'First question')
  assert.match(calls[1][1].content, /some or all may not have been heard/)
  assert.ok(calls[1][1].content.endsWith('Available response text'))
  assert.equal(calls[1][2].content, 'Follow-up')
})

test('failed synthesis before playback does not leave a dangling user in the next request', async () => {
  const calls = []
  let transcripts = 0
  const { VoiceSession } = await moduleAt('../src/audio/voiceSession.ts', {
    '../api/voice': {
      transcribeVoice: async () => ++transcripts === 1 ? 'Failed question' : 'Retry question',
      streamVoice: async (_tier, _backend, messages, _signal, onEvent) => {
        calls.push(messages)
        await onEvent({ type: 'text', text: 'Generated but unheard' })
        throw new Error('Synthesis failed')
      },
    },
    './voiceAudio': audio, './voiceHistory': history,
  })
  const session = new VoiceSession(capabilities, 'current', 0.02, () => {})
  session.context = { sampleRate: 16000 }
  session.player = { playbackBegan: false, stop() {} }
  await session.respond(new Float32Array(10), performance.now())
  await session.respond(new Float32Array(10), performance.now())
  assert.deepEqual(calls[1], [{ role: 'user', content: 'Retry question' }])
})
