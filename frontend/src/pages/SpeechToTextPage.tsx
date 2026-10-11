import { useEffect, useRef, useState } from 'react'
import { Panel } from '../components/Controls'
import { recordingToWav, requestTranscription } from '../api/transcription'

/** Choose or record audio locally and submit only after pressing Submit. */
export default function SpeechToTextPage() {
  const [recording, setRecording] = useState(false)
  const [pending, setPending] = useState(false)
  const [clip, setClip] = useState<Blob | null>(null)
  const [text, setText] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [seconds, setSeconds] = useState(0)
  const recorder = useRef<MediaRecorder | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const controller = useRef<AbortController | null>(null)
  const busy = useRef(false)
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; controller.current?.abort(); stream.current?.getTracks().forEach(track => track.stop()) }
  }, [])
  useEffect(() => {
    if (!recording) return
    const timer = window.setInterval(() => setSeconds(value => value + 1), 1000)
    return () => window.clearInterval(timer)
  }, [recording])
  async function toggleRecording() {
    if (recorder.current?.state === 'recording') { recorder.current.stop(); return }
    if (busy.current) return
    busy.current = true
    setPending(true); setError('')
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (!mounted.current) { media.getTracks().forEach(track => track.stop()); return }
      stream.current = media
      const instance = new MediaRecorder(media)
      recorder.current = instance
      const chunks: Blob[] = []
      instance.ondataavailable = event => { if (event.data.size) chunks.push(event.data) }
      instance.onstop = () => {
        media.getTracks().forEach(track => track.stop())
        if (mounted.current) { setClip(new Blob(chunks, { type: instance.mimeType })); setRecording(false) }
      }
      instance.start(); setRecording(true); setSeconds(0); setClip(null); setText(null)
    } catch (reason) { stream.current?.getTracks().forEach(track => track.stop()); setError(reason instanceof Error ? reason.message : 'Cannot record audio.') }
    finally { busy.current = false; if (mounted.current) setPending(false) }
  }
  async function submit() {
    if (!clip || busy.current) return
    busy.current = true; setPending(true); setError('')
    const abort = new AbortController(); controller.current = abort
    try {
      const audio = await recordingToWav(clip)
      if (abort.signal.aborted) return
      const transcript = await requestTranscription(audio, false, abort.signal)
      if (!abort.signal.aborted) setText(transcript)
    } catch (reason) { if (!abort.signal.aborted) setError(reason instanceof Error ? reason.message : 'Transcription failed.') }
    finally { busy.current = false; if (mounted.current) setPending(false) }
  }
  return <div className="scroll-page"><div className="recording-layout"><Panel className="recording-panel">
    <button type="button" className="record-button" disabled={pending} aria-label={recording ? 'Stop recording' : 'Start recording'} aria-pressed={recording} onClick={() => void toggleRecording()}><span /></button>
    <span className="recording-timer">{Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, '0')}.0</span>
    <label>Audio file
      <input type="file" accept="audio/*" disabled={recording || pending} onChange={event => {
        const file = event.target.files?.[0]
        if (!file) return
        setClip(file); setText(null); setError(''); setSeconds(0)
        event.target.value = ''
      }} />
    </label>
    {clip instanceof File && <p>{clip.name}</p>}
    <div className="row">
      <button type="button" className="button" disabled={recording || pending || !clip} onClick={() => { setClip(null); setText(null); setSeconds(0); setError('') }}>Discard</button>
      <button type="button" className="button button--muted" disabled={recording || pending || !clip} onClick={() => void submit()}>{pending ? 'Working…' : 'Submit'}</button>
    </div>
    {error && <p role="alert" className="error">{error}</p>}
    {text !== null && <p role="status">{text || 'No speech detected.'}</p>}
  </Panel></div></div>
}
