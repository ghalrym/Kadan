import { useEffect, useRef, useState } from 'react'
import { Panel } from '../components/Controls'
import type { TranscriptionResponse } from '../api/generated'
import { recordingToWav, requestTranscription } from '../api/transcription'

type Phase = 'idle' | 'starting' | 'recording' | 'ready' | 'transcribing'

function formatTime(total: number) {
  const seconds = Math.max(0, Math.floor(total))
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}

const hints: Record<Phase, string> = {
  idle: 'Click to start recording',
  starting: 'Waiting for microphone…',
  recording: 'Recording — click to stop',
  ready: 'Click to record again',
  transcribing: 'Transcribing…',
}

/** Choose or record audio locally and submit only after pressing Transcribe. */
export default function SpeechToTextPage() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [clip, setClip] = useState<Blob | null>(null)
  const [clipUrl, setClipUrl] = useState('')
  const [duration, setDuration] = useState(0)
  const [text, setText] = useState<TranscriptionResponse | null>(null)
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [copied, setCopied] = useState(false)
  const [copyError, setCopyError] = useState('')
  const copyGeneration = useRef(0)
  const recorder = useRef<MediaRecorder | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const controller = useRef<AbortController | null>(null)
  const mounted = useRef(true)
  const recording = phase === 'recording'
  const busy = phase === 'starting' || phase === 'transcribing'

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      controller.current?.abort()
      if (recorder.current?.state === 'recording') recorder.current.stop()
      stream.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  useEffect(() => {
    if (!recording) return
    const started = performance.now()
    const timer = window.setInterval(
      () => setElapsed((performance.now() - started) / 1000),
      200,
    )
    return () => window.clearInterval(timer)
  }, [recording])

  useEffect(
    () => () => {
      if (clipUrl) URL.revokeObjectURL(clipUrl)
    },
    [clipUrl],
  )

  function clearCopyFeedback() {
    copyGeneration.current++
    setCopied(false)
    setCopyError('')
  }

  function selectClip(next: Blob | null, length = 0) {
    clearCopyFeedback()
    setClip(next)
    setClipUrl(next ? URL.createObjectURL(next) : '')
    setDuration(length)
    setText(null)
    setError('')
    setPhase(next ? 'ready' : 'idle')
  }

  async function toggleRecording() {
    if (recorder.current?.state === 'recording') {
      recorder.current.stop()
      return
    }
    if (busy) return
    setPhase('starting')
    setError('')
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (!mounted.current) {
        media.getTracks().forEach((track) => track.stop())
        return
      }
      stream.current = media
      const instance = new MediaRecorder(media)
      recorder.current = instance
      const chunks: Blob[] = []
      const started = performance.now()
      instance.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data)
      }
      instance.onstop = () => {
        media.getTracks().forEach((track) => track.stop())
        stream.current = null
        recorder.current = null
        if (!mounted.current) return
        const length = (performance.now() - started) / 1000
        selectClip(
          chunks.length ? new Blob(chunks, { type: instance.mimeType }) : null,
          length,
        )
        if (!chunks.length) setError('No audio was captured.')
      }
      instance.start()
      selectClip(null)
      setElapsed(0)
      setPhase('recording')
    } catch (reason) {
      stream.current?.getTracks().forEach((track) => track.stop())
      stream.current = null
      if (!mounted.current) return
      setPhase(clip ? 'ready' : 'idle')
      setError(
        reason instanceof Error ? reason.message : 'Cannot record audio.',
      )
    }
  }

  async function submit() {
    if (!clip || busy || recording) return
    clearCopyFeedback()
    setPhase('transcribing')
    setError('')
    const abort = new AbortController()
    controller.current = abort
    try {
      const audio = await recordingToWav(clip)
      if (abort.signal.aborted) return
      const transcript = await requestTranscription(audio, abort.signal)
      if (!abort.signal.aborted) {
        clearCopyFeedback()
        setText(transcript)
      }
    } catch (reason) {
      if (!abort.signal.aborted)
        setError(
          reason instanceof Error ? reason.message : 'Transcription failed.',
        )
    } finally {
      if (controller.current === abort) {
        controller.current = null
        if (mounted.current) setPhase('ready')
      }
    }
  }

  function cancel() {
    controller.current?.abort()
    controller.current = null
    setPhase(clip ? 'ready' : 'idle')
  }

  async function copy() {
    if (!text?.text) return
    clearCopyFeedback()
    const generation = copyGeneration.current
    try {
      await navigator.clipboard.writeText(text.text)
      if (!mounted.current || generation !== copyGeneration.current) return
      setCopied(true)
      window.setTimeout(() => {
        if (mounted.current && generation === copyGeneration.current)
          setCopied(false)
      }, 1500)
    } catch {
      if (mounted.current && generation === copyGeneration.current)
        setCopyError('Could not copy to the clipboard.')
    }
  }

  const fileName = clip instanceof File ? clip.name : 'Recording'
  const shownTime = recording ? elapsed : duration
  const transcript =
    text?.text ||
    (text?.formatting_status === 'formatted'
      ? 'No text after formatting.'
      : 'No speech detected.')

  return (
    <div className="scroll-page">
      <div className="recording-layout">
        <Panel className="recording-panel">
          <div className="recorder">
            <button
              type="button"
              className={`record-button${recording ? ' record-button--active' : ''}`}
              disabled={busy}
              aria-label={recording ? 'Stop recording' : 'Start recording'}
              aria-pressed={recording}
              onClick={() => void toggleRecording()}
            >
              <span />
            </button>
            <span
              className={`recording-timer${recording ? ' recording-timer--live' : ''}`}
              aria-live="off"
            >
              {formatTime(shownTime)}
            </span>
            <span className="recording-hint" aria-live="polite">
              {hints[phase]}
            </span>
          </div>

          <div className="recording-divider" aria-hidden="true">
            <span>or</span>
          </div>

          <label
            className={`audio-drop${recording || busy ? ' audio-drop--disabled' : ''}`}
          >
            <input
              type="file"
              accept="audio/*"
              className="sr-only"
              disabled={recording || busy}
              onChange={(event) => {
                const file = event.target.files?.[0]
                event.target.value = ''
                if (file) selectClip(file)
              }}
            />
            <span className="audio-drop-title">Choose an audio file</span>
            <span className="faint">WAV, MP3, M4A, OGG, WebM…</span>
          </label>

          {clip && clipUrl && (
            <div className="clip-preview">
              <div className="clip-preview-header">
                <span className="clip-name" title={fileName}>
                  {fileName}
                </span>
                {duration > 0 && (
                  <span className="mono faint">{formatTime(duration)}</span>
                )}
              </div>
              <audio
                controls
                src={clipUrl}
                onLoadedMetadata={(event) => {
                  const length = event.currentTarget.duration
                  if (Number.isFinite(length)) setDuration(length)
                }}
              />
            </div>
          )}

          <div className="recording-actions">
            <button
              type="button"
              className="button"
              disabled={recording || busy || !clip}
              onClick={() => selectClip(null)}
            >
              Discard
            </button>
            {phase === 'transcribing' ? (
              <button type="button" className="button" onClick={cancel}>
                <span className="recording-spinner" aria-hidden="true" />
                Cancel
              </button>
            ) : (
              <button
                type="button"
                className="button button--primary"
                disabled={recording || busy || !clip}
                onClick={() => void submit()}
              >
                Transcribe
              </button>
            )}
          </div>

          {error && (
            <p role="alert" className="recording-error">
              {error}
            </p>
          )}
        </Panel>

        {text !== null && (
          <Panel className="transcript-panel">
            <div className="panel-header">
              <span className="section-heading">Transcript</span>
              {text.language && (
                <span className="eyebrow">{text.language}</span>
              )}
              <button
                type="button"
                className="button"
                disabled={!text.text}
                onClick={() => void copy()}
              >
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
            <div className="panel-body stack">
              {text.formatting_status === 'failed' && (
                <p role="alert" className="recording-error">
                  Formatting failed: {text.formatting_error} Raw transcription
                  shown.
                </p>
              )}
              {copyError && (
                <p role="alert" className="recording-error">
                  {copyError}
                </p>
              )}
              <p
                role="status"
                className={`transcript-text${text.text ? '' : ' faint'}`}
              >
                {transcript}
              </p>
              {text.formatting_status === 'formatted' && (
                <details className="transcript-raw">
                  <summary className="eyebrow">Raw transcription</summary>
                  <p className="transcript-text muted">{text.raw_text}</p>
                </details>
              )}
            </div>
          </Panel>
        )}
      </div>
    </div>
  )
}
