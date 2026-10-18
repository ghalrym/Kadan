/** Local energy VAD, not speech recognition. Keeps 300ms pre-roll and bounded turns. */
export class VoiceActivity {
  private frames: Float32Array[] = []
  private active = false
  private hot = 0
  private quiet = 0
  private duration = 0
  readonly sampleRate: number
  readonly threshold: number
  constructor(sampleRate: number, threshold = 0.02) {
    this.sampleRate = sampleRate
    this.threshold = threshold
  }
  snapshot(): Float32Array | null {
    if (!this.active) return null
    const samples = new Float32Array(this.frames.reduce((sum, frame) => sum + frame.length, 0))
    let offset = 0
    for (const frame of this.frames) { samples.set(frame, offset); offset += frame.length }
    return samples
  }
  push(frame: Float32Array): { started: boolean; ended?: Float32Array; level: number; truncated: boolean } {
    const seconds = frame.length / this.sampleRate
    const level = Math.sqrt(frame.reduce((sum, x) => sum + x * x, 0) / frame.length)
    const voiced = level >= this.threshold
    this.hot = voiced ? this.hot + seconds : 0
    this.quiet = voiced ? 0 : this.quiet + seconds
    this.frames.push(frame)
    let started = false
    if (!this.active && this.hot >= 0.06) {
      this.active = true
      this.duration = this.frames.reduce((sum, x) => sum + x.length, 0) / this.sampleRate
      started = true
    }
    if (!this.active) {
      while (this.frames.length > Math.ceil(0.3 / seconds)) this.frames.shift()
      return { started, level, truncated: false }
    }
    this.duration += seconds
    const truncated = this.duration >= 20
    if (this.quiet >= 0.6 || truncated) {
      const ended = new Float32Array(this.frames.reduce((sum, x) => sum + x.length, 0))
      let offset = 0
      for (const samples of this.frames) { ended.set(samples, offset); offset += samples.length }
      this.frames = []; this.active = false; this.duration = 0; this.hot = 0; this.quiet = 0
      return { started, ended, level, truncated }
    }
    return { started, level, truncated: false }
  }
}

export function wavDataUrl(samples: Float32Array, sampleRate: number): string {
  const bytes = new Uint8Array(44 + samples.length * 2)
  const view = new DataView(bytes.buffer)
  const word = (offset: number, value: string) => { for (let i = 0; i < value.length; i++) bytes[offset + i] = value.charCodeAt(i) }
  word(0, 'RIFF'); view.setUint32(4, bytes.length - 8, true); word(8, 'WAVEfmt ')
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); word(36, 'data'); view.setUint32(40, samples.length * 2, true)
  samples.forEach((value, index) => view.setInt16(44 + index * 2, Math.round(Math.max(-1, Math.min(1, value)) * (value < 0 ? 32768 : 32767)), true))
  let binary = ''
  for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192))
  return `data:audio/wav;base64,${btoa(binary)}`
}

/** Keep recognition windows within the API limit regardless of device capture rate. */
export function recognitionWav(samples: Float32Array, sampleRate: number): string {
  if (sampleRate === 16000) return wavDataUrl(samples, sampleRate)
  const ratio = sampleRate / 16000
  const resampled = new Float32Array(Math.floor(samples.length / ratio))
  for (let i = 0; i < resampled.length; i++) {
    const start = i * ratio, end = (i + 1) * ratio
    let value = 0
    for (let j = Math.floor(start); j < Math.ceil(end); j++)
      value += samples[j] * (Math.min(end, j + 1) - Math.max(start, j))
    resampled[i] = value / ratio
  }
  return wavDataUrl(resampled, 16000)
}

function waitForPlayback(signal: AbortSignal): Promise<void> {
  signal.throwIfAborted()
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(signal.reason) }
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve() }, 25)
    signal.addEventListener('abort', abort, { once: true })
  })
}

/** Schedules PCM against the audio clock, retaining at most five seconds. */
export class VoicePlayer {
  private context: AudioContext
  private sources = new Set<AudioBufferSourceNode>()
  private nextTime = 0
  private started = false
  private firstStartTime: number | null = null
  private generation = 0
  seconds = 0
  underruns = 0
  gapMs = 0
  firstScheduledAt: number | null = null
  constructor(context: AudioContext) { this.context = context }
  get playing() { return this.sources.size > 0 }
  get playbackBegan() { return this.firstStartTime !== null && this.context.currentTime >= this.firstStartTime }
  async push(encoded: string, sampleRate: number, signal: AbortSignal) {
    const generation = this.generation
    const binary = atob(encoded)
    if (!binary.length || binary.length % 2 || binary.length > sampleRate * 2 * 5)
      throw new Error('Invalid PCM chunk size; the API must send at most five seconds per event.')
    const duration = binary.length / 2 / sampleRate
    while (this.nextTime - this.context.currentTime + duration > 5) await waitForPlayback(signal)
    signal.throwIfAborted()
    if (generation !== this.generation) throw new DOMException('Playback cancelled', 'AbortError')
    const bytes = Uint8Array.from(binary, value => value.charCodeAt(0))
    const view = new DataView(bytes.buffer)
    const buffer = this.context.createBuffer(1, binary.length / 2, sampleRate)
    const channel = buffer.getChannelData(0)
    for (let i = 0; i < channel.length; i++) channel[i] = view.getInt16(i * 2, true) / 32768
    const now = this.context.currentTime
    const start = Math.max(now + 0.04, this.nextTime)
    if (this.started && start > this.nextTime + 0.02) {
      this.underruns++; this.gapMs += (start - this.nextTime) * 1000
    }
    if (!this.started) {
      this.firstStartTime = start
      this.firstScheduledAt = performance.now() + (start - now + this.context.baseLatency + (this.context.outputLatency || 0)) * 1000
    }
    this.started = true
    this.nextTime = start + duration
    this.seconds += duration
    const source = this.context.createBufferSource()
    source.buffer = buffer; source.connect(this.context.destination)
    this.sources.add(source)
    source.onended = () => { this.sources.delete(source); source.disconnect() }
    source.start(start)
  }
  async drain(signal: AbortSignal) {
    while (this.playing) await waitForPlayback(signal)
  }
  stop() {
    this.generation++
    for (const source of this.sources) { source.onended = null; source.stop(); source.disconnect() }
    this.sources.clear(); this.nextTime = 0; this.started = false; this.firstStartTime = null
    this.seconds = 0; this.underruns = 0; this.gapMs = 0; this.firstScheduledAt = null
  }
}
