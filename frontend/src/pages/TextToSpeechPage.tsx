import { useEffect, useRef, useState } from 'react'
import { ModeNavigation } from '../components/Controls'
import { AudioCard } from '../components/Media'
import { speechScript, voiceDescription } from '../data/playground'
import { fetchSpeechHistory, requestSpeech, speechRequest } from '../api/speech'
import { listSpeechModels } from '../api/generated/sdk.gen'
import type {
  GeneratedSpeech,
  SpeechModelOption,
} from '../api/generated/types.gen'

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
 * Preserve the editor layout while accepting local samples and complete WAV results.
 */
function SpeechWorkspace({ clone }: { clone: boolean }) {
  const [script, setScript] = useState(speechScript)
  const [description, setDescription] = useState(voiceDescription)
  const [sample, setSample] = useState('')
  const [transcript, setTranscript] = useState('')
  const [speakerOnly, setSpeakerOnly] = useState(false)
  const [speaker, setSpeaker] = useState('Ryan')
  const [model, setModel] = useState('')
  const [models, setModels] = useState<SpeechModelOption[]>([])
  const custom = model.endsWith('-custom')
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
        const choices = result.data.filter((item) =>
          clone ? item.mode === 'clone' : item.mode !== 'clone',
        )
        setModels(choices)
        setModel(choices[0]?.id ?? '')
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setError(String(failure))
      })
    return () => controller.abort()
  }, [clone])

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
   * Keep user input for retry, expose API errors, and append only validated audio
   * from the still-active controller; cancellation cannot create a local audio result.
   */
  async function submit() {
    if (active.current || loadingHistory) return
    setError(null)
    setNotice('')
    let body
    try {
      body = custom
        ? {
            script: script.trim(),
            model_id: model,
            voice: {
              mode: 'custom' as const,
              speaker,
              instruction: model === 'qwen-tts-1.7b-custom' ? description : '',
            },
          }
        : speechRequest(
            script,
            clone ? 'clone' : 'describe',
            clone ? sample : description,
          )
      body.model_id = model
      if (body.voice.mode === 'clone') {
        body.voice.transcript = transcript
        body.voice.speaker_only = speakerOnly
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
          {custom && (
            <div className="field">
              <label htmlFor="speech-speaker" className="eyebrow">
                Speaker
              </label>
              <select
                id="speech-speaker"
                className="input"
                value={speaker}
                disabled={pending}
                onChange={(event) => setSpeaker(event.target.value)}
              >
                {[
                  'Vivian',
                  'Serena',
                  'Uncle_Fu',
                  'Dylan',
                  'Eric',
                  'Ryan',
                  'Aiden',
                  'Ono_Anna',
                  'Sohee',
                ].map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </div>
          )}
          {clone ? (
            <div className="field">
              <label htmlFor="speech-sample" className="eyebrow">
                Upload a voice sample
              </label>
              <input
                id="speech-sample"
                type="file"
                accept="audio/*"
                disabled={pending}
                onChange={async (event) => {
                  const file = event.target.files?.[0]
                  setSample('')
                  if (!file) return
                  try {
                    const bytes = new Uint8Array(await file.arrayBuffer())
                    let value = ''
                    for (const byte of bytes) value += String.fromCharCode(byte)
                    setSample(btoa(value))
                  } catch {
                    setError('Could not read the voice sample.')
                  }
                }}
              />
              <label htmlFor="speech-reference" className="eyebrow">
                Reference transcript
              </label>
              <textarea
                id="speech-reference"
                className="input"
                value={transcript}
                disabled={pending || speakerOnly}
                onChange={(event) => setTranscript(event.target.value)}
              />
              <label>
                <input
                  type="checkbox"
                  checked={speakerOnly}
                  disabled={pending}
                  onChange={(event) => setSpeakerOnly(event.target.checked)}
                />{' '}
                Speaker only
              </label>
            </div>
          ) : model !== 'qwen-tts-0.6b-custom' ? (
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
          ) : null}
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
                !script.trim() ||
                (clone
                  ? !sample || (!speakerOnly && !transcript.trim())
                  : !custom && !description.trim())
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
