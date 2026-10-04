import { useEffect, useRef, useState } from 'react'
import { ModeNavigation } from '../components/Controls'
import { fetchSpeechHistory, requestSpeech, speechRequest } from '../api/speech'
import type { GeneratedSpeech } from '../api/generated/types.gen'

export default function TextToSpeechPage({
  clone = false,
}: {
  clone?: boolean
}) {
  return <SpeechWorkspace key={clone ? 'clone' : 'describe'} clone={clone} />
}

function SpeechWorkspace({ clone }: { clone: boolean }) {
  const [script, setScript] = useState('')
  const [description, setDescription] = useState('')
  const [sample, setSample] = useState('')
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

  function cancel() {
    active.current?.abort()
    active.current = null
    setPending(false)
    setNotice('Request cancelled in this browser.')
  }

  async function submit() {
    if (active.current || loadingHistory) return
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
        <p className="muted">
          No speech provider is configured. You can edit and submit settings to
          test API validation; audio generation is unavailable.
        </p>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault()
            void submit()
          }}
        >
          {clone ? (
            <div className="field">
              <label htmlFor="speech-sample" className="eyebrow">
                Voice sample reference
              </label>
              <input
                id="speech-sample"
                className="input"
                value={sample}
                maxLength={2000}
                disabled={pending}
                onChange={(event) => setSample(event.target.value)}
                aria-describedby="sample-help"
              />
              <p id="sample-help" className="faint">
                Reference text only. No file upload service exists, and the
                backend does not fetch this reference.
              </p>
            </div>
          ) : (
            <div className="field">
              <label htmlFor="speech-description" className="eyebrow">
                Voice description
              </label>
              <textarea
                id="speech-description"
                className="input"
                rows={3}
                value={description}
                maxLength={2000}
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
              maxLength={5000}
              disabled={pending}
              onChange={(event) => setScript(event.target.value)}
            />
            <span className="mono faint character-count">
              {script.length} / 5,000
            </span>
          </div>
          {error && <p role="alert">{error}</p>}
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
                !script.trim() || !(clone ? sample : description).trim()
              }
            >
              {error ? 'Retry speech request' : 'Generate speech'}
            </button>
          )}
        </form>
      </aside>
      <div className="workspace-results stack" aria-busy={loadingHistory}>
        <h2 className="eyebrow">Speech history</h2>
        {loadingHistory && <p role="status">Loading speech history…</p>}
        {historyError && (
          <>
            <p role="alert">{historyError}</p>
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
          <p className="muted">No generated speech.</p>
        )}
        {audio.map((item, index) => (
          <article className="panel" key={index}>
            <h3>{item.voice}</h3>
            <p>{item.script}</p>
            <p className="faint">
              {item.meta} · {item.time}
            </p>
          </article>
        ))}
      </div>
    </div>
  )
}
