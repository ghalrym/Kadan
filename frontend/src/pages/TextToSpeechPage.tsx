import { useEffect, useRef, useState } from 'react'
import { ModeNavigation, UploadPlaceholder } from '../components/Controls'
import { AudioCard } from '../components/Media'
import { speechScript, voiceDescription } from '../data/playground'
import { fetchSpeechHistory, requestSpeech, speechRequest } from '../api/speech'
import type { GeneratedSpeech } from '../api/generated/types.gen'

/**
 * Choose describe/clone mode and reset workspace state when the mode changes.
 */
export default function TextToSpeechPage({
  clone = false,
}: {
  clone?: boolean
}) {
  return <SpeechWorkspace key={clone ? 'clone' : 'describe'} clone={clone} />
}

/**
 * Own editable speech fields, cancellable history and one active submission.
 * Preserve the original upload and audio layout with disabled media controls
 * until a transport and playable files are available. Unmount aborts reads and invalidates submissions.
 */
function SpeechWorkspace({ clone }: { clone: boolean }) {
  const [script, setScript] = useState(speechScript)
  const [description, setDescription] = useState(voiceDescription)
  const sample = '' // Await the upload transport; keep the original placeholder.
  const [audio, setAudio] = useState<GeneratedSpeech[]>([])
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [refresh, setRefresh] = useState(0)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState('')
  const active = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    void fetchSpeechHistory(controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) setAudio(items)
      })
      .catch((failure) => {
        if (!controller.signal.aborted)
          setHistoryError(
            failure instanceof Error
              ? failure.message
              : 'Could not load speech history.',
          )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingHistory(false)
      })
    return () => controller.abort()
  }, [refresh])

  useEffect(() => {
    return () => {
      active.current?.abort()
      active.current = null
    }
  }, [])

  /**
   * Abort the browser request, release pending UI state and reject late results.
   */
  function cancel() {
    active.current?.abort()
    active.current = null
    setPending(false)
    setNotice('Request cancelled in this browser.')
  }

  /**
   * Validate the selected voice mode and submit once history loading has finished.
   * Keep user input for retry, expose API errors, and append only validated metadata
   * from the still-active controller; cancellation cannot create a local audio result.
   */
  async function submit() {
    if (active.current || loadingHistory || clone) return
    setError(null)
    setNotice('')
    let body
    try {
      body = speechRequest(
        script,
        clone ? 'clone' : 'describe',
        clone ? sample : description,
      )
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Check your input.')
      return
    }
    const controller = new AbortController()
    active.current = controller
    setPending(true)
    try {
      const result = await requestSpeech(body, controller.signal)
      if (active.current === controller && !controller.signal.aborted) {
        setAudio((items) => [result, ...items])
        setNotice(
          'Speech response received. This API contract does not include a playable media URL.',
        )
      }
    } catch (failure) {
      if (active.current === controller && !controller.signal.aborted)
        setError(
          failure instanceof Error ? failure.message : 'Speech request failed.',
        )
    } finally {
      if (active.current === controller) {
        active.current = null
        setPending(false)
      }
    }
  }

  return (
    <div className="workspace workspace--speech">
      <aside className="workspace-controls" aria-label="Speech settings">
        <div className="field">
          <span className="eyebrow">Voice</span>
          <ModeNavigation
            label="Voice mode"
            options={[
              { label: 'Describe a voice', to: '/tts' },
              { label: 'Clone a voice', to: '/tts/clone' },
            ]}
          />
        </div>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault()
            void submit()
          }}
        >
          {clone ? (
            <UploadPlaceholder title="Upload a voice sample" caption="WAV or MP3 · 10–30 s of clean speech" />
          ) : (
            <div className="field">
              <textarea
                id="speech-description"
                aria-label="Voice description"
                className="input"
                rows={3}
                value={description}
                disabled={pending}
                onChange={(event) => setDescription(event.target.value)}
              />
            </div>
          )}
          <div className="speech-script field">
            <label htmlFor="speech-script" className="eyebrow">
              Script
            </label>
            <textarea
              id="speech-script"
              className="input"
              rows={6}
              value={script}
              disabled={pending}
              onChange={(event) => setScript(event.target.value)}
            />
            <span className="mono faint character-count">
              {script.length} characters
            </span>
          </div>
          {error && <p role="alert" className="error-panel">{error}</p>}
          {notice && <p role="status">{notice}</p>}
          {pending ? (
            <>
              <p role="status">Waiting for speech API…</p>
              <button
                type="button"
                className="button button--secondary"
                onClick={cancel}
              >
                Cancel
              </button>
            </>
          ) : (
            <button
              type="submit"
              className="button button--primary"
              disabled={
                loadingHistory || clone ||
                !script.trim() ||
                !(clone ? sample : description).trim()
              }
            >
              {error ? 'Retry speech request' : 'Generate speech'}
            </button>
          )}
        </form>
      </aside>
      <div className="workspace-results stack" aria-busy={loadingHistory}>
        <h2 className="eyebrow">Generated audio</h2>
        {loadingHistory && <p role="status">Loading speech history…</p>}
        {historyError && (
          <>
            <p role="alert" className="error-panel">{historyError}</p>
            <button
              type="button"
              className="button button--secondary"
              disabled={pending}
              onClick={() => {
                setLoadingHistory(true)
                setHistoryError(null)
                setRefresh((value) => value + 1)
              }}
            >
              Retry history
            </button>
          </>
        )}
        {!loadingHistory && !historyError && audio.length === 0 && (
          <AudioCard voice="No generated audio" meta="—" script="" time="—" />
        )}
        {audio.map((item, index) => (
          <AudioCard key={index} voice={item.voice} meta={item.meta} script={item.script} time={item.time} />
        ))}
      </div>
    </div>
  )
}
