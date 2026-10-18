import { useEffect, useRef, useState } from 'react'
import { voiceCapabilities } from '../api/voice'
import type { VoiceBackend, VoiceCapabilities } from '../api/voice'
import { microphoneUnavailable, VoiceSession } from '../audio/voiceSession'
import type { VoiceMeasurement, VoiceState } from '../audio/voiceSession'
import './VoicePage.css'

type RecordedTurn = VoiceMeasurement & { session: number }
const milliseconds = (value: number | null) => value === null ? '—' : `${Math.round(value)} ms`

export default function VoicePage() {
  const [capabilities, setCapabilities] = useState<VoiceCapabilities | null>(null)
  const [loading, setLoading] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [backend, setBackend] = useState<VoiceBackend>('current')
  const [threshold, setThreshold] = useState(0.02)
  const [running, setRunning] = useState(false)
  const [state, setState] = useState<VoiceState>({ status: 'Stopped', partial: '', level: 0, error: null, microphone: '', measurements: [] })
  const [records, setRecords] = useState<RecordedTurn[]>([])
  const active = useRef<VoiceSession | null>(null)
  const sessionNumber = useRef(0)
  const selected = capabilities?.backends.find(item => item.id === backend)
  const browserUnavailable = microphoneUnavailable()

  useEffect(() => {
    const controller = new AbortController()
    void voiceCapabilities(controller.signal).then(result => {
      if (!controller.signal.aborted) setCapabilities(result)
    }).catch(failure => {
      if (!controller.signal.aborted) { setCapabilities(null); setError(failure instanceof Error ? failure.message : String(failure)) }
    }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [refresh])

  useEffect(() => () => { const session = active.current; active.current = null; session?.stop() }, [])

  function stop() { active.current?.stop(); active.current = null; setRunning(false) }

  async function start() {
    if (!capabilities || active.current) return
    setError(null)
    const sessionId = ++sessionNumber.current
    const session = new VoiceSession(capabilities, backend, threshold, snapshot => {
      if (active.current !== session) return
      setState(snapshot)
      setRecords(existing => [...snapshot.measurements.map(item => ({ ...item, session: sessionId })), ...existing.filter(item => item.session !== sessionId)].slice(0, 100))
      if (snapshot.status === 'Stopped') { active.current = null; setRunning(false) }
    })
    active.current = session
    setState({ status: 'Starting microphone…', partial: '', level: 0, error: null, microphone: '', measurements: [] })
    setRunning(true)
    try { await session.start() }
    catch (failure) {
      if (active.current === session) {
        setError(failure instanceof Error ? failure.message : String(failure))
        stop()
      }
    }
  }

  function exportMeasurements() {
    const blob = new Blob([JSON.stringify({ exportedAt: new Date().toISOString(), browser: navigator.userAgent, capabilities,
      definitions: { firstPcmMs: 'Turn POST to first PCM received; excludes ASR.', firstScheduledAudioMs: 'Turn POST to estimated speaker output (audio schedule plus reported device latency). Not measured acoustic output.', endOfSpeechToAudioMs: 'Estimated VAD speech end to scheduled speaker output; includes endpoint wait and ASR.', receiveRtf: 'Turn POST to last PCM scheduled / generated audio duration; includes chat, queueing, transport and playback backpressure. Not isolated model RTF.', gapMs: 'Audio clock gaps over 20ms after playback began.', serverTiming: 'Worker-reported timing; see backend documentation.' }, turns: records }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a'); link.href = url; link.download = `hikari-voice-${Date.now()}.json`; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const latest = records[0]
  return (
    <div className="workspace voice-workspace">
      <aside className="workspace-controls stack" aria-label="Voice settings">
        <div className="field">
          <label htmlFor="voice-backend">Speech backend</label>
          <select id="voice-backend" className="input" value={backend} disabled={running} onChange={event => setBackend(event.target.value as VoiceBackend)}>
            <option value="current">Current tier TTS</option>
            <option value="drama3">Drama 3 FP8 · experimental</option>
            <option value="drama3-int4">Drama 3 INT4 · unsupported</option>
          </select>
        </div>
        <dl className="voice-models">
          <dt>Active tier</dt><dd>{capabilities?.tier ?? 'Unavailable'}</dd>
          <dt>Recognition</dt><dd>{capabilities?.asr_model ?? 'Unavailable'}</dd>
          <dt>Conversation</dt><dd>{capabilities?.chat_model ?? 'Unavailable'}</dd>
          <dt>Speech</dt><dd>{selected?.speech_model ?? 'Unavailable'}</dd>
          <dt>Speech delivery</dt><dd>{selected ? selected.streaming ? 'Streamed PCM' : 'Buffered synthesis, chunked playback' : 'Unavailable'}</dd>
          <dt>Scheduling</dt><dd>{capabilities?.scheduling ?? 'Unavailable'}</dd>
        </dl>
        {loading && <p role="status">Checking worker capabilities…</p>}
        {!loading && capabilities && (!capabilities.available || !selected?.available) && <p className="error-panel" role="status">{selected?.reason ?? capabilities.reason ?? 'This backend is unavailable.'}</p>}
        {browserUnavailable && <p className="error-panel">{browserUnavailable}</p>}
        {error && <p className="error-panel" role="alert">{error}</p>}
        <button type="button" className="button button--secondary" disabled={running || loading} onClick={() => { setLoading(true); setError(null); setRefresh(value => value + 1) }}>Refresh capabilities</button>
        <div className="field">
          <label htmlFor="voice-threshold">Barge-in sensitivity · {threshold.toFixed(3)} RMS</label>
          <input id="voice-threshold" type="range" min="0.005" max="0.08" step="0.005" value={threshold} disabled={running} onChange={event => setThreshold(Number(event.target.value))} />
          <span className="muted">Lower values detect quieter speech and more background noise.</span>
        </div>
        <button type="button" className="button button--primary" disabled={!running && (loading || !!browserUnavailable || !capabilities?.available || !selected?.available)} onClick={() => running ? stop() : void start()}>{running ? 'Stop conversation' : 'Start conversation'}</button>
        <p className="muted">Use headphones. Browser echo cancellation is requested but cannot guarantee that speaker output will not trigger barge-in.</p>
        <details className="voice-details">
          <summary>Capture and hardware limits</summary>
          <p>Microphone access requires HTTPS or localhost, including on a LAN. Browser permissions and network security are unchanged.</p>
          <p>Microphone capture continues during playback. Recognition revises a cumulative window every 2 seconds; it is not native streaming ASR. A 600 ms pause ends a turn. Turns are limited to 20 seconds.</p>
          <p>Speech onset stops local playback and cancels the request. Server cancellation waits for a cooperative inference boundary. The single worker serializes recognition, conversation and speech; listening does not imply concurrent GPU inference.</p>
          <p>Starting pins the active tier and models. Stop and refresh after changing tier settings. Each start creates a new conversation.</p>
          <p>Context retains recent user/assistant pairs within 40 messages and 32000 UTF-8 bytes, dropping oldest pairs when needed. Interrupted playback retains generated response text marked as potentially unheard. Turns that fail before playback begins are omitted from context.</p>
          {capabilities?.notes.map(note => <p key={note}>{note}</p>)}
        </details>
      </aside>
      <div className="workspace-results stack voice-results">
        <div className="voice-status">
          <strong role="status">{state.status}</strong>
          <meter min="0" max="0.15" value={state.level} aria-label="Microphone RMS level" />
          <span className="muted">{state.microphone || 'Microphone off'}</span>
        </div>
        {state.error && <p className="error-panel" role="alert">{state.error}</p>}
        {state.partial && <p className="voice-partial"><span className="muted">Interim · may be revised</span>{state.partial}</p>}
        <dl className="voice-metrics">
          <div><dt>First PCM</dt><dd>{milliseconds(latest?.firstPcmMs ?? null)}</dd></div>
          <div><dt>First audio · estimated</dt><dd>{milliseconds(latest?.firstScheduledAudioMs ?? null)}</dd></div>
          <div><dt>Speech end → audio</dt><dd>{milliseconds(latest?.endOfSpeechToAudioMs ?? null)}</dd></div>
          <div><dt>Speech RTF · server</dt><dd>{latest?.serverTiming?.speech_rtf == null ? '—' : `${latest.serverTiming.speech_rtf.toFixed(2)}×`}</dd></div>
          <div><dt>Playback gaps</dt><dd>{latest ? `${latest.underruns} · ${Math.round(latest.gapMs)} ms` : '—'}</dd></div>
          <div><dt>Interrupted turns</dt><dd>{records.filter(item => item.status === 'interrupted').length}</dd></div>
        </dl>
        <div className="voice-actions">
          <button type="button" className="button button--secondary" disabled={!records.length} onClick={exportMeasurements}>Export measurements</button>
          <button type="button" className="button button--text" disabled={running || !records.length} onClick={() => setRecords([])}>Clear results</button>
        </div>
        <details className="voice-details">
          <summary>Measurement definitions</summary>
          <p>First PCM and first audio start at the conversation request, after recognition. First audio estimates scheduled output plus reported device latency; it is not an acoustic measurement. Speech end → audio includes endpoint delay and recognition.</p>
          <p>Speech RTF is server speech time divided by audio duration, including queueing, loading and stream backpressure. Export also includes client receive RTF (conversation request time / audio duration) and native worker timings when available. Gaps count scheduled audio clock gaps over 20 ms.</p>
          <p>Results stay in this tab. Export retains up to 100 turns, including transcripts. No GPU performance result is implied until you run a conversation.</p>
        </details>
        {!records.length && <p className="muted">No conversation measurements yet.</p>}
        <ol className="voice-turns" aria-label="Conversation turns">
          {records.map(item => <li key={`${item.session}-${item.id}`}>
            <div className="voice-turn-meta"><span>{item.backend} · {item.tier}</span><span>{item.status}</span></div>
            <p><span className="muted">You</span>{item.transcript || 'Transcribing…'}</p>
            {item.response && <p><span className="muted">Hikari</span>{item.response}</p>}
            <div className="voice-turn-meta"><span>ASR {milliseconds(item.asrMs)}</span><span>Audio {item.audioSeconds.toFixed(2)} s</span><span>Speech end → audio {milliseconds(item.endOfSpeechToAudioMs)}</span></div>
          </li>)}
        </ol>
      </div>
    </div>
  )
}
