import { useEffect, useRef, useState } from 'react'
import { Panel, SectionHeading } from '../components/Controls'
import { modelSettings } from '../data/playground'
import type { ModelsResponse, CheckpointStatus, ModelLifecycleStatus, SettingsModels } from '../api/generated'
import { configureContextV1ModelsModelIdContextPut, updateSettings } from '../api/generated/sdk.gen'
import './SettingsPage.css'
import { WhisperSelector } from '../components/WhisperSelector'
import { ModelPicker } from '../components/ModelPicker'

/**
 * Request catalog status or a mutation under /v1/models with a 15-second timeout.
 * Combines caller cancellation with the timeout; throws backend detail when
 * available, or an HTTP error for non-JSON proxy failures. Returns typed status.
 */
async function requestCheckpointStatus(path = '', method: 'GET' | 'POST' | 'DELETE' = 'GET', body?: { license_acknowledged: boolean }, signal?: AbortSignal): Promise<ModelsResponse> {
  const response = await fetch(`/v1/models${path}`, {
    method, signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!response.ok) {
    let message = `Model service returned HTTP ${response.status}`
    try {
      const errorResponse: unknown = await response.json()
      if (typeof errorResponse === 'object' && errorResponse !== null && 'detail' in errorResponse && typeof errorResponse.detail === 'string') {
        message = errorResponse.detail
      }
    } catch { /* A proxy error may not be JSON. */ }
    throw new Error(message)
  }
  return response.json()
}
/** Read runtime state; requests load the selected model on demand, so Settings never loads one. */
async function requestLifecycle(signal: AbortSignal): Promise<ModelLifecycleStatus> {
  const response = await fetch('/model-lifecycle', { signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]) })
  const result = await response.json()
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : `Model status failed (HTTP ${response.status})`)
  return result
}

/** Format bytes as decimal GB to match the upstream checkpoint size estimates. */
const formatGigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`

const contextPresets = [
  ['8k', 8192], ['16k', 16384], ['32k', 32768], ['64k', 65536],
  ['128k', 131072], ['500k', 500000], ['1M', 1000000],
] as const

/**
 * Persist a context preset immediately; the next request loads with it.
 * Known architecture bounds disable oversized choices; an unknown bound is not
 * guessed. A saved non-preset limit stays selectable.
 */
function ContextControl({ model, pending, change }: { model: CheckpointStatus; pending: boolean; change: (limit: number | null) => void }) {
  const value = model.context_limit
  const customLimits = value !== null && !contextPresets.some(([, preset]) => preset === value) ? [value] : []
  return <div className="field model-context">
    <label htmlFor={`context-${model.id}`}>Max context length (tokens)</label>
    <select id={`context-${model.id}`} className="input" value={value?.toString() ?? ''} disabled={pending} onChange={event => change(event.target.value === '' ? null : Number(event.target.value))}>
      {contextPresets.map(([label, limit]) => <option key={limit} value={limit} disabled={model.architecture_context_limit !== null && limit > model.architecture_context_limit}>{label}</option>)}
      {customLimits.map(limit => <option key={limit} value={limit} disabled={model.architecture_context_limit !== null && limit > model.architecture_context_limit}>{limit} (saved)</option>)}
      <option value="">Architecture maximum</option>
    </select>
  </div>
}

function ModelDownloadStatus({ model, pending, downloading, cancel, retry }: {
  model: CheckpointStatus; pending: boolean; downloading: boolean;
  cancel: (model: CheckpointStatus) => void; retry: (model: CheckpointStatus) => void;
}) {
  const active = model.status === 'downloading' || model.status === 'cancelling'
  return <div className="model-download-status model-video-progress">
    <div className="model-download-heading">
      <p role="status">{model.display_name} · {model.status}</p>
      <button type="button" className="button" disabled={pending || model.status === 'cancelling' || (downloading && model.status !== 'downloading')} onClick={() => {
        if (model.status === 'downloading') cancel(model)
        else retry(model)
      }}>{model.status === 'cancelling' ? 'Cancelling…' : model.status === 'downloading' ? 'Cancel download' : 'Retry download'}</button>
    </div>
    {active && <>
      <progress className="progress" aria-label={`${model.id} download progress`} max={model.total_bytes || 1} value={model.total_bytes ? model.downloaded_bytes : undefined} />
      <span className="mono faint">{formatGigabytes(model.downloaded_bytes)} / {model.total_bytes ? formatGigabytes(model.total_bytes) : 'checking checkpoint size'}</span>
    </>}
    {model.error && <p role="alert" className="error">{model.error}</p>}
  </div>
}

/** Format a token count with the preset label when one matches, else a grouped number. */
const formatContext = (limit: number) => contextPresets.find(([, preset]) => preset === limit)?.[0] ?? limit.toLocaleString()

/** Preserve the original modality icons alongside the compact settings rows. */
function ModelIcon({ type }: { type: string }) {
  return <svg className={`model-icon type-${type}`} width="13" height="13" viewBox="0 0 12 12" aria-hidden="true">
    <circle cx="6" cy="3" r="2.5" /><circle cx="8.85" cy="5.07" r="2.5" />
    <circle cx="7.76" cy="8.43" r="2.5" /><circle cx="4.24" cy="8.43" r="2.5" />
    <circle cx="3.15" cy="5.07" r="2.5" /><circle className="model-icon-center" cx="6" cy="6" r="1.5" />
  </svg>
}

function LicenseNotice({ model, close, proceed }: { model: CheckpointStatus; close: () => void; proceed: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog className="model-license-dialog" ref={dialog} aria-labelledby="model-license-title" onCancel={event => { event.preventDefault(); close() }}>
    <h2 id="model-license-title">{model.display_name ?? model.repo_id} license</h2>
    <p>{model.license_notice} {model.license_url && <a href={model.license_url} target="_blank" rel="noreferrer">License</a>}</p>
    <div className="model-license-actions"><button type="button" className="button" autoFocus onClick={close}>Cancel</button><button type="button" className="button button--primary" onClick={proceed}>Continue</button></div>
  </dialog>
}

/**
 * Show live downloads and persist model/context choices; requests load on demand.
 * Polling and mutations have separate error state; request generations prevent
 * older polls from overwriting mutations. Unmount aborts outstanding requests.
 */
export default function SettingsPage() {
  const mutationVersion = useRef(0)
  const actionController = useRef<AbortController | null>(null)
  useEffect(() => () => { actionController.current?.abort() }, [])
  const [modelStatus, setCheckpointStatus] = useState<ModelsResponse | null>(null)
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const [pending, setPending] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [speechId, setSpeechId] = useState<string | null>(null)
  const [licenseModel, setLicenseModel] = useState<CheckpointStatus | null>(null)
  const licenseAction = useRef(false)
  const [videoId, setVideoId] = useState<string | null>(null)
  const [imageError, setImageError] = useState('')
  const [lifecycle, setLifecycle] = useState<ModelLifecycleStatus | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    /**
     * Refresh status only if no mutation superseded this request. Poll again
     * after completion, and stop updates/timers when the effect is disposed.
     */
    async function poll() {
      const version = mutationVersion.current
      try {
        const [next, runtime] = await Promise.all([requestCheckpointStatus('', 'GET', undefined, controller.signal), requestLifecycle(controller.signal)])
        if (!controller.signal.aborted && !actionController.current && version === mutationVersion.current) { setCheckpointStatus(next); setLifecycle(runtime); setError('') }
      } catch (failure) {
        if (!controller.signal.aborted && !actionController.current && version === mutationVersion.current) {
          setLifecycle(null)
          setError(failure instanceof Error ? failure.message : 'Could not reach model service')
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1500)
      }
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [refresh])

  /** Mutation versions prevent older polls from overwriting an action's result. */
  async function mutateModel(action: (signal: AbortSignal) => Promise<void>, reportError = setActionError) {
    if (actionController.current) return
    const controller = new AbortController()
    actionController.current = controller
    mutationVersion.current += 1
    setPending(true)
    reportError('')
    try {
      await action(controller.signal)
    } catch (failure) {
      if (!controller.signal.aborted) reportError(failure instanceof Error ? failure.message : 'Model request failed')
    } finally {
      mutationVersion.current += 1
      actionController.current = null
      if (!controller.signal.aborted) setPending(false)
    }
  }

  function submitModelChange(path: string, method: 'POST' | 'DELETE', body?: { license_acknowledged: boolean }) {
    return mutateModel(async signal => {
      const next = await requestCheckpointStatus(path, method, body, signal)
      if (!signal.aborted) { setCheckpointStatus(next); setError('') }
    })
  }

  function saveContext(modelId: string, contextLimit: number | null) {
    return mutateModel(async signal => {
      const result = await configureContextV1ModelsModelIdContextPut({
        path: { model_id: modelId },
        body: { context_limit: contextLimit },
        signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]),
      })
      if (!result.response?.ok || !result.data) {
        const detail: unknown = (result.error as { detail?: unknown } | undefined)?.detail
        throw new Error(typeof detail === 'string' ? detail : result.response ? `Context save failed (HTTP ${result.response.status})` : 'Could not reach model service')
      }
      if (!signal.aborted) { setCheckpointStatus(result.data); setError('') }
    })
  }

  /**
   * Persist a selection. Choosing a different language model also unloads the
   * resident one server-side, which can wait behind the queue head.
   */
  function selectModel(type: 'LLM' | 'Image', model: CheckpointStatus, reportError: typeof setActionError) {
    return mutateModel(async signal => {
      const models: SettingsModels = type === 'LLM' ? { LLM: model.id as SettingsModels['LLM'] } : { Image: model.id as SettingsModels['Image'] }
      const result = await updateSettings({ body: { models }, signal: AbortSignal.any([signal, AbortSignal.timeout(type === 'LLM' ? 60000 : 15000)]) })
      if (!result.response?.ok) {
        const detail: unknown = (result.error as { detail?: unknown } | undefined)?.detail
        throw new Error(typeof detail === 'string' ? detail : result.response ? `${type === 'LLM' ? 'Language' : 'Image'} model selection failed (HTTP ${result.response.status})` : 'Could not reach model service')
      }
      const [next, runtime] = await Promise.all([requestCheckpointStatus('', 'GET', undefined, signal), requestLifecycle(signal)])
      if (!signal.aborted) {
        setCheckpointStatus(next)
        setLifecycle(runtime)
        setError('')
      }
    }, reportError)
  }
  /** Require a fresh explicit acknowledgement for each licensed download or retry. */
  function downloadModel(model: CheckpointStatus) {
    if (actionController.current) return
    if (model.license_notice) { licenseAction.current = false; setLicenseModel(model) }
    else void submitModelChange(`/${model.id}/download`, 'POST')
  }
  const models = modelStatus?.models ?? []
  const imageModels = models.filter(model => model.kind === 'image')
  const image = imageModels.find(model => model.id === modelStatus?.selected_image)
  const imageDownload = imageModels.find(model => ['downloading', 'cancelling', 'failed', 'cancelled'].includes(model.status))
  const videoModels = models.filter(model => model.kind === 'video')
  const video = videoModels.find(model => model.id === videoId) ?? videoModels[0]
  const videoDownload = videoModels.find(model => ['downloading', 'cancelling', 'failed', 'cancelled'].includes(model.status))
  const speechModels = models.filter(model => model.kind === 'speech')
  const speech = speechModels.find(model => model.id === speechId) ?? speechModels[0]
  const speechDownload = speechModels.find(model => ['downloading', 'cancelling', 'failed', 'cancelled'].includes(model.status))

  const languageModels = models.filter(model => !model.kind || model.kind === 'llm')
  const downloading = models.some(model => model.status === 'downloading' || model.status === 'cancelling')
  const chosen = languageModels.find(model => model.id === modelStatus?.selected_model_id)
  const runtimeModel = models.find(model => model.id === lifecycle?.model_id)?.repo_id.split('/')[1] ?? lifecycle?.model_id
  const runtimeContext = lifecycle?.effective_context_limit ?? lifecycle?.configured_context_limit
  const runtime: { tone: string; text: string } =
    !lifecycle ? { tone: 'idle', text: modelStatus || error ? 'Status unavailable' : 'Checking status…' }
    : lifecycle.state === 'loading' ? { tone: 'busy', text: `Loading ${runtimeModel ?? 'model'}…` }
    : lifecycle.state === 'unloading' ? { tone: 'busy', text: `Unloading ${runtimeModel ?? 'model'}…` }
    : lifecycle.state === 'ready' || lifecycle.state === 'offloaded' ? { tone: lifecycle.state === 'ready' ? 'ready' : 'idle', text: `${runtimeModel} ${lifecycle.state === 'ready' ? 'running' : 'offloaded'}${runtimeContext ? ` · ${formatContext(runtimeContext)} context` : ''}` }
    : lifecycle.state === 'error' ? { tone: 'error', text: 'Load failed' }
    : { tone: 'idle', text: 'No model loaded' }
  const downloadStatus = languageModels.find(model => model.status === 'downloading' || model.status === 'cancelling')
    ?? languageModels.find(model => ['cancelled', 'failed'].includes(model.status))

  return (
    <div className="scroll-page">
      {licenseModel && <LicenseNotice model={licenseModel} close={() => { setLicenseModel(null); document.getElementById(`model-${licenseModel.kind === 'video' ? 'Video' : licenseModel.kind === 'image' ? 'Image' : 'LLM'}`)?.focus() }} proceed={() => {
        if (licenseAction.current) return
        licenseAction.current = true
        const model = licenseModel
        setLicenseModel(null)
        void submitModelChange(`/${model.id}/download`, 'POST', { license_acknowledged: true })
      }} />}
      <div className="settings-layout stack">
        <SectionHeading>Models</SectionHeading>
        <Panel className="model-settings">
          <div className="model-row model-row--llm">
            <div className="model-llm-header">
              <div className="model-llm-title">
                <label htmlFor="model-LLM"><ModelIcon type="LLM" />Language model</label>
                <p className={`model-runtime model-runtime--${runtime.tone}`} aria-live="polite"><span className="model-runtime-dot" aria-hidden="true" />{runtime.text}</p>
              </div>
            </div>
            <div className="model-llm-fields">
              <div className="field">
                <span className="field-label" aria-hidden="true">Model</span>
                <ModelPicker models={languageModels} current={chosen} selectedId={chosen?.id ?? null} pending={pending} downloading={downloading}
                  choose={model => void selectModel('LLM', model, setActionError)}
                  download={downloadModel} />
              </div>
              {chosen
                ? <ContextControl model={chosen} pending={pending} change={limit => void saveContext(chosen.id, limit)} />
                : <div className="field model-context"><label htmlFor="context-unselected">Max context length (tokens)</label><select id="context-unselected" className="input" disabled><option>Choose a language model</option></select></div>}
            </div>
            <div className="model-llm-details stack compact">
              {!modelStatus && !error && <p role="status" className="muted">Loading model catalog…</p>}
              {error && <div role="alert" className="error-panel model-alert"><span>{error}</span><button type="button" className="button" onClick={() => setRefresh(value => value + 1)}>Refresh</button></div>}
              {(actionError || lifecycle?.error) && <p role="alert" className="error-panel">{actionError || lifecycle?.error}</p>}
              {(downloadStatus ? [downloadStatus] : []).map(model => {
                const active = model.status === 'downloading' || model.status === 'cancelling'
                return <div className="model-download-status" key={model.id}>
                  <div className="model-download-heading">
                    <p role="status"><span>{(model.display_name ?? model.repo_id.split('/')[1])}</span> · <span className="muted">{model.status}</span></p>
                    <button type="button" className="button" disabled={pending || model.status === 'cancelling' || (!active && downloading)} onClick={() => { if (active) void submitModelChange(`/${model.id}/download`, 'DELETE'); else downloadModel(model) }}>{active ? model.status === 'cancelling' ? 'Cancelling…' : 'Cancel download' : 'Retry download'}</button>
                  </div>
                  {active && <><progress className="progress" aria-label={`${model.id} download progress`} max={model.total_bytes || 1} value={model.total_bytes ? model.downloaded_bytes : undefined} /><span className="mono faint">{formatGigabytes(model.downloaded_bytes)} / {model.total_bytes ? formatGigabytes(model.total_bytes) : 'checking checkpoint size'}</span></>}
                  {model.error && <p role="alert" className="error">{model.error}</p>}
                </div>
              })}
            </div>
          </div>
          {modelSettings.filter(model => model.type !== 'LLM').map(model => model.type === 'TTS' ? <div className="model-row model-row--video" key={model.type}>
            <label htmlFor="model-TTS"><ModelIcon type="TTS" />{model.label}</label>
            <div className="field"><ModelPicker pickerId="TTS" label="TTS model" models={speechModels} current={speech} selectedId={speechId} pending={pending} downloading={downloading} choose={model => setSpeechId(model.id)} download={model => { setSpeechId(model.id); downloadModel(model) }} /></div>
            {speech && !speech.inference_available && <span className="muted model-speech-note">Download only</span>}
            {speechDownload && <ModelDownloadStatus model={speechDownload} pending={pending} downloading={downloading}
              cancel={model => void submitModelChange(`/${model.id}/download`, 'DELETE')} retry={downloadModel} />}
          </div> : model.type === 'Image' ? <div className="model-row model-row--video" key={model.type}>
            <label htmlFor="model-Image"><ModelIcon type="Image" />{model.label}</label>
            <div className="field"><ModelPicker pickerId="Image" label="Image model" models={imageModels} current={image} selectedId={image?.id ?? null} pending={pending} downloading={downloading} choose={model => void selectModel('Image', model, setImageError)} download={downloadModel} /></div>
            {image && image.status !== 'complete' && <span className="muted model-video-note">Not downloaded</span>}
            {imageError && <p role="alert" className="error-panel model-video-progress">{imageError}</p>}
            {imageDownload && <ModelDownloadStatus model={imageDownload} pending={pending} downloading={downloading}
              cancel={model => void submitModelChange(`/${model.id}/download`, 'DELETE')} retry={downloadModel} />}
          </div> : model.type === 'Video' ? <div className="model-row model-row--video" key={model.type}>
            <label htmlFor="model-Video"><ModelIcon type="Video" />{model.label}</label>
            <div className="field"><ModelPicker pickerId="Video" label="Video model" models={videoModels} current={video} selectedId={videoId} pending={pending} downloading={downloading} choose={model => setVideoId(model.id)} download={model => { setVideoId(model.id); downloadModel(model) }} /></div>
            {video && !video.inference_available && <span className="muted model-video-note">Download only</span>}
            {videoDownload && <ModelDownloadStatus model={videoDownload} pending={pending} downloading={downloading}
              cancel={model => void submitModelChange(`/${model.id}/download`, 'DELETE')} retry={downloadModel} />}
          </div> : <div className="model-row" key={model.type}>
            <label htmlFor={`model-${model.type}`}><ModelIcon type={model.type} />{model.label}</label>
            {model.type === 'STT' ? <WhisperSelector models={models.filter(item => item.kind === 'transcription')} pending={pending} downloading={downloading} download={downloadModel} cancel={model => void submitModelChange(`/${model.id}/download`, 'DELETE')} /> : <select className="input" id={`model-${model.type}`} value={model.selected} disabled>
              {model.options.map(option => <option key={option}>{option}</option>)}
            </select>}
          </div>)}
          <div className="model-row">
            <span className="model-label"><ModelIcon type="STT" />Whisper S1 Mini formatting</span>
            <button type="button" role="switch" aria-checked="true" aria-label="Whisper S1 Mini formatting" disabled className="switch"><span /></button>
          </div>
        </Panel>
      </div>
    </div>
  )
}
