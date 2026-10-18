import { streamVoice, transcribeVoice } from '../api/voice'
import type { VoiceBackend, VoiceCapabilities, VoiceTiming } from '../api/voice'
import { VoiceActivity, VoicePlayer, recognitionWav } from './voiceAudio'
import { VoiceHistory } from './voiceHistory'

export type VoiceMeasurement = {
  id: number
  backend: VoiceBackend
  tier: string
  speechModel: string | null
  startedAt: string
  status: 'running' | 'complete' | 'interrupted' | 'failed'
  transcript: string
  response: string
  asrMs: number | null
  firstPcmMs: number | null
  firstScheduledAudioMs: number | null
  endOfSpeechToAudioMs: number | null
  audioSeconds: number
  receiveRtf: number | null
  underruns: number
  gapMs: number
  serverTiming?: VoiceTiming
}
export type VoiceState = { status: string; partial: string; level: number; error: string | null; microphone: string; measurements: VoiceMeasurement[] }

export function microphoneUnavailable(): string | null {
  if (!window.isSecureContext) return 'Microphone access needs HTTPS or localhost. A plain HTTP LAN address is not a secure context.'
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === 'undefined')
    return 'This browser does not support microphone capture with AudioWorklet.'
  return null
}

/** One browser session pins the tier and invalidates every in-flight callback on interruption. */
export class VoiceSession {
  private capabilities: VoiceCapabilities
  private backend: VoiceBackend
  private publish: (state: VoiceState) => void
  private state: VoiceState = { status: 'Starting microphone…', partial: '', level: 0, error: null, microphone: '', measurements: [] }
  private context: AudioContext | null = null
  private stream: MediaStream | null = null
  private capture: AudioWorkletNode | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private player: VoicePlayer | null = null
  private activity: VoiceActivity | null = null
  private turn: AbortController | null = null
  private interim: AbortController | null = null
  private generation = 0
  private stopped = false
  private lastPartial = 0
  private lastLevel = 0
  private history = new VoiceHistory()
  private measurement: VoiceMeasurement | null = null
  private sequence = 0
  private threshold: number

  constructor(capabilities: VoiceCapabilities, backend: VoiceBackend, threshold: number, publish: (state: VoiceState) => void) {
    this.capabilities = capabilities; this.backend = backend; this.threshold = threshold; this.publish = publish
  }
  private update(values: Partial<VoiceState>) {
    if (this.stopped) return
    this.state = { ...this.state, ...values }
    this.publish(this.state)
  }
  private record() {
    if (!this.measurement) return
    const measurement = { ...this.measurement }
    this.update({ measurements: [measurement, ...this.state.measurements.filter(item => item.id !== measurement.id)].slice(0, 50) })
  }
  async start() {
    const unavailable = microphoneUnavailable()
    if (unavailable) throw new Error(unavailable)
    if (!this.capabilities.available || !this.capabilities.tier || !this.capabilities.asr_model || !this.capabilities.backends.find(item => item.id === this.backend)?.available)
      throw new Error('The selected voice backend is unavailable. Refresh capabilities before starting.')
    // Resume within the user gesture, before awaiting microphone permission.
    const context = new AudioContext()
    this.context = context
    await context.resume()
    if (this.stopped) return
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
    if (this.stopped) { stream.getTracks().forEach(track => track.stop()); return }
    this.stream = stream
    const track = stream.getAudioTracks()[0]
    track.onended = () => { if (!this.stopped) { this.update({ error: 'Microphone disconnected. Stop, reconnect it, then start again.' }); this.stop() } }
    const settings = track.getSettings()
    this.update({ microphone: `Echo cancellation: ${settings.echoCancellation === undefined ? 'not reported' : settings.echoCancellation ? 'on' : 'off'} · ${context.sampleRate} Hz capture` })
    await context.audioWorklet.addModule(new URL('./voiceCapture.worklet.js?no-inline', import.meta.url))
    if (this.stopped) return
    this.activity = new VoiceActivity(context.sampleRate, this.threshold)
    this.player = new VoicePlayer(context)
    this.source = context.createMediaStreamSource(stream)
    this.capture = new AudioWorkletNode(context, 'voice-capture')
    this.capture.onprocessorerror = () => { this.update({ error: 'Microphone processing stopped. Stop and restart the conversation.' }); this.stop() }
    this.capture.port.onmessage = event => this.frame(event.data as Float32Array)
    this.source.connect(this.capture)
    this.capture.connect(context.destination)
    context.onstatechange = () => {
      if (!this.stopped && context.state !== 'running') {
        this.update({ error: 'Browser audio was suspended. Start a new conversation to resume microphone and playback.' })
        this.stop()
      }
    }
    this.update({ status: 'Listening' })
  }
  private frame(samples: Float32Array) {
    if (this.stopped || !this.activity) return
    const now = performance.now()
    const result = this.activity.push(samples)
    if (now - this.lastLevel > 100) { this.lastLevel = now; this.update({ level: result.level }) }
    if (result.started) {
      this.interrupt()
      this.lastPartial = now
      this.update({ status: 'Listening · speech detected', partial: '', error: null })
    }
    if (result.ended) {
      this.interim?.abort(); this.interim = null
      // Endpoint delay is excluded from estimated acoustic end, retained in E2E latency.
      void this.respond(result.ended, now - (result.truncated ? 0 : 600))
    } else if (now - this.lastPartial >= 2000 && !this.interim) {
      const snapshot = this.activity.snapshot()
      if (snapshot) { this.lastPartial = now; void this.partial(snapshot) }
    }
  }
  private async partial(samples: Float32Array) {
    const controller = new AbortController()
    this.interim = controller
    const generation = this.generation
    try {
      const text = await transcribeVoice(recognitionWav(samples, this.context!.sampleRate), this.capabilities.tier!, this.capabilities.asr_model!, controller.signal)
      if (!controller.signal.aborted && generation === this.generation) this.update({ partial: text })
    } catch (error) {
      if (!controller.signal.aborted) this.update({ error: `Interim recognition failed: ${error instanceof Error ? error.message : String(error)}` })
    } finally { if (this.interim === controller) this.interim = null }
  }
  private interrupt() {
    this.generation++
    this.turn?.abort(); this.turn = null
    this.interim?.abort(); this.interim = null
    if (this.measurement?.status === 'running') {
      if (this.player?.playbackBegan) this.history.commit(this.measurement.transcript, this.measurement.response, true)
      this.measurement.status = 'interrupted'; this.record()
    }
    this.player?.stop()
  }
  private async respond(samples: Float32Array, speechEndedAt: number) {
    this.interrupt()
    const generation = this.generation
    const controller = new AbortController()
    this.turn = controller
    const began = performance.now()
    const measurement: VoiceMeasurement = {
      id: ++this.sequence, backend: this.backend, tier: this.capabilities.tier!,
      speechModel: this.capabilities.backends.find(item => item.id === this.backend)!.speech_model,
      startedAt: new Date().toISOString(), status: 'running', transcript: '', response: '',
      asrMs: null, firstPcmMs: null, firstScheduledAudioMs: null, endOfSpeechToAudioMs: null,
      audioSeconds: 0, receiveRtf: null, underruns: 0, gapMs: 0,
    }
    this.measurement = measurement
    this.record()
    this.update({ status: 'Transcribing · microphone listening', partial: '' })
    try {
      const transcript = await transcribeVoice(recognitionWav(samples, this.context!.sampleRate), this.capabilities.tier!, this.capabilities.asr_model!, controller.signal)
      controller.signal.throwIfAborted()
      if (generation !== this.generation) return
      measurement.asrMs = performance.now() - began
      measurement.transcript = transcript
      if (!transcript) { measurement.status = 'complete'; this.record(); this.update({ status: 'Listening · no words recognized' }); return }
      const messages = this.history.request(transcript)
      const requestedAt = performance.now()
      let firstReceivedAt: number | null = null
      this.update({ status: 'Generating · microphone listening' })
      await streamVoice(this.capabilities.tier!, this.backend, messages, controller.signal, async event => {
        if (generation !== this.generation || controller.signal.aborted) return
        if (event.type === 'text') measurement.response += event.text
        if (event.type === 'audio') {
          const received = performance.now()
          if (firstReceivedAt === null) { firstReceivedAt = received; measurement.firstPcmMs = received - requestedAt }
          await this.player!.push(event.audio, event.sample_rate, controller.signal)
          if (generation !== this.generation) return
          measurement.audioSeconds = this.player!.seconds
          measurement.underruns = this.player!.underruns
          measurement.gapMs = this.player!.gapMs
          measurement.firstScheduledAudioMs = this.player!.firstScheduledAt! - requestedAt
          measurement.endOfSpeechToAudioMs = this.player!.firstScheduledAt! - speechEndedAt
          measurement.receiveRtf = (performance.now() - requestedAt) / 1000 / measurement.audioSeconds
          this.update({ status: 'Playing · microphone listening' })
        }
        if (event.type === 'done') measurement.serverTiming = event.timing
        this.record()
      })
      controller.signal.throwIfAborted()
      if (!measurement.audioSeconds) throw new Error('The backend completed without speech audio.')
      await this.player!.drain(controller.signal)
      controller.signal.throwIfAborted()
      if (generation !== this.generation) return
      this.history.commit(transcript, measurement.response, false)
      measurement.status = 'complete'; this.record(); this.update({ status: 'Listening' })
    } catch (error) {
      if (!controller.signal.aborted && generation === this.generation) {
        if (this.player?.playbackBegan) this.history.commit(measurement.transcript, measurement.response, true)
        measurement.status = 'failed'; this.record(); this.player?.stop()
        this.update({ status: 'Listening · last turn failed', error: error instanceof Error ? error.message : String(error) })
      }
    } finally { if (this.turn === controller) this.turn = null }
  }
  stop() {
    if (this.stopped) return
    this.interrupt()
    this.update({ status: 'Stopped', level: 0, partial: '' })
    this.stopped = true
    if (this.capture) { this.capture.port.onmessage = null; this.capture.port.close(); this.capture.disconnect() }
    this.source?.disconnect()
    this.stream?.getTracks().forEach(track => { track.onended = null; track.stop() })
    if (this.context) this.context.onstatechange = null
    void this.context?.close().catch(() => {})
  }
}
