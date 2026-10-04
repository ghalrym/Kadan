import { useEffect, useRef, useState } from 'react'
import { Panel } from '../components/Controls'
import { MAX_AUDIO_REFERENCE_LENGTH, requestTranscription } from '../api/transcription'

export default function SpeechToTextPage() {
  const [audio, setAudio] = useState('')
  const [formatting, setFormatting] = useState(true)
  const [pending, setPending] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const active = useRef<AbortController | null>(null)
  useEffect(() => () => { active.current?.abort(); active.current = null }, [])

  async function submit() {
    if (active.current || !audio.trim()) return
    const controller = new AbortController()
    active.current = controller
    setPending(true)
    setTranscript('')
    setError('')
    setNotice('')
    try {
      const text = await requestTranscription(audio, formatting, AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]))
      if (active.current === controller && !controller.signal.aborted) setTranscript(text)
    } catch (failure) {
      if (active.current === controller && !controller.signal.aborted) {
        setError(failure instanceof Error ? failure.message : 'Transcription request failed.')
      }
    } finally {
      if (active.current === controller) { active.current = null; setPending(false) }
    }
  }

  function cancel() {
    active.current?.abort()
    active.current = null
    setPending(false)
    setNotice('Request cancelled in this browser.')
  }

  async function copy() {
    try { await navigator.clipboard.writeText(transcript); setNotice('Transcript copied.') }
    catch { setError('Could not copy. Select the transcript and copy it manually.') }
  }

  return (
    <div className="scroll-page">
      <div className="settings-layout stack">
        <Panel className="recording-panel">
          <h2>Speech to text</h2>
          <p>No speech-to-text provider is configured yet. Submit an audio reference to test the API connection; it currently reports unavailable. Recording, file upload, and fetching audio references are not implemented.</p>
          <form className="stack" onSubmit={event => { event.preventDefault(); void submit() }}>
            <label htmlFor="audio-reference">Audio reference</label>
            <input id="audio-reference" className="input" value={audio} maxLength={MAX_AUDIO_REFERENCE_LENGTH} disabled={pending} onChange={event => setAudio(event.target.value)} placeholder="Audio reference for the future provider" />
            <label><input type="checkbox" checked={formatting} disabled={pending} onChange={event => setFormatting(event.target.checked)} /> Format transcript</label>
            <div className="row">
              <button type="submit" className="button button--primary" disabled={pending || !audio.trim()}>{pending ? 'Submitting…' : 'Submit'}</button>
              {pending && <button type="button" className="button button--secondary" onClick={cancel}>Cancel</button>}
            </div>
          </form>
          {pending && <p role="status">Waiting for the transcription API…</p>}
          {error && <p role="alert">{error}</p>}
          {notice && <p role="status">{notice}</p>}
          {transcript && <div className="stack"><label htmlFor="transcript">Transcript</label><textarea id="transcript" className="input" rows={8} readOnly value={transcript} /><button type="button" className="button button--secondary" onClick={() => void copy()}>Copy transcript</button></div>}
        </Panel>
      </div>
    </div>
  )
}
