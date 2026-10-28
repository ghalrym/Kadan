import type { SpeechRequest } from '../api/generated/types.gen'
import { useEffect, useRef, useState } from 'react'
import { AudioCard } from '../components/Media'
import { speechScript } from '../data/playground'
import { fetchSpeechHistory, requestSpeech } from '../api/speech'
import { listSpeechModels } from '../api/generated/sdk.gen'
import type {
  GeneratedSpeech,
  SpeechModelOption,
} from '../api/generated/types.gen'

/** Own supported LFM speech choices, cancellable history and one submission. */
export default function TextToSpeechPage() {
  const [script, setScript] = useState(speechScript)
  const [language, setLanguage] = useState<NonNullable<SpeechRequest['language']>>('American English')
  const [gender, setGender] = useState<NonNullable<SpeechRequest['gender']>>('female')
  const [model, setModel] = useState('')
  const [models, setModels] = useState<SpeechModelOption[]>([])
  const selectedModel = models.find((item) => item.id === model)
  const chosenLanguage = selectedModel?.languages.includes(language)
    ? language : selectedModel?.default_language ?? 'American English'
  const chosenGender = selectedModel?.genders.includes(gender)
    ? gender : selectedModel?.default_gender ?? 'female'
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
    void listSpeechModels({ signal: controller.signal })
      .then((result) => {
        if (controller.signal.aborted) return
        if (!result.response?.ok || !result.data)
          throw new Error('Could not load speech models.')
        const choices = result.data
        setModels(choices)
        setModel(choices[0]?.id ?? '')
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setError(String(failure))
      })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void fetchSpeechHistory(controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) setAudio((current) => [...current, ...items])
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
   * Validate the selected LFM speech choices after history loading has finished.
   * Keep user input for retry, expose API errors, and append only validated audio
   * from the still-active controller; cancellation cannot create a local audio result.
   */
  async function submit() {
    if (active.current || loadingHistory) return
    setError(null)
    setNotice('')
    let body: SpeechRequest
    try {
      if (model !== 'lfm3-audio-2b-bf16')
        throw new Error('Select the LFM3 Audio speech model.')
      body = {
        script: script.trim(), model_id: model,
        language: chosenLanguage, gender: chosenGender,
      }
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
        setNotice('Speech generated.')
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
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault()
            void submit()
          }}
        >
          <div className="field">
            <label htmlFor="speech-model" className="eyebrow">
              Model
            </label>
            <select
              id="speech-model"
              className="input"
              value={model}
              disabled={pending}
              onChange={(event) => setModel(event.target.value)}
            >
              {!models.length && (
                <option value="">No speech models available</option>
              )}
              {models.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </div>
          {selectedModel && <>
            <div className="field">
              <label htmlFor="speech-language" className="eyebrow">Language / locale</label>
              <select id="speech-language" className="input" value={chosenLanguage}
                disabled={pending} onChange={(event) => setLanguage(event.target.value as NonNullable<SpeechRequest['language']>)}>
                {selectedModel.languages.map((name) => <option key={name}>{name}</option>)}
              </select>
            </div>
            <div className="field">
              <label htmlFor="speech-gender" className="eyebrow">Voice gender</label>
              <select id="speech-gender" className="input" value={chosenGender}
                disabled={pending} onChange={(event) => setGender(event.target.value as NonNullable<SpeechRequest['gender']>)}>
                {selectedModel.genders.map((name) => <option key={name} value={name}>{name === 'female' ? 'Female' : 'Male'}</option>)}
              </select>
            </div>
            <p className="faint">LFM3 Audio · 24 kHz WAV · up to 60 seconds.</p>
          </>}
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
          {error && (
            <p role="alert" className="error-panel">
              {error}
            </p>
          )}
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
                loadingHistory ||
                !model ||
                !script.trim() || !chosenLanguage || !chosenGender
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
            <p role="alert" className="error-panel">
              {historyError}
            </p>
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
          <AudioCard
            key={index}
            voice={item.voice}
            meta={item.meta}
            script={item.script}
            time={item.time}
            audioUrl={
              item.audio_base64
                ? `data:audio/wav;base64,${item.audio_base64}`
                : undefined
            }
          />
        ))}
      </div>
    </div>
  )
}
