import { useEffect, useRef, useState } from 'react'
import { Panel, SectionHeading } from '../components/Controls'
import { modelSettings } from '../data/playground'
import type { ModelsResponse, ModelStatus, ModelLifecycleStatus } from '../api/generated'
import './SettingsPage.css'

/**
 * Request catalog status or a mutation under /v1/models with a 15-second timeout.
 * Combines caller cancellation with the timeout; throws backend detail when
 * available, or an HTTP error for non-JSON proxy failures. Returns typed status.
 */
async function requestModelStatus(path = '', method: 'GET' | 'POST' | 'DELETE' = 'GET', body?: { license_acknowledged: boolean }, signal?: AbortSignal): Promise<ModelsResponse> {
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
/** Send one lifecycle request; only the server can confirm readiness after load. */
async function requestLifecycle(signal: AbortSignal, body?: { model_id: string; context_limit: number | null }): Promise<ModelLifecycleStatus> {
  const response = await fetch(`/model-lifecycle${body ? '/load' : ''}`, {
    method: body ? 'POST' : 'GET',
    signal: AbortSignal.any([signal, AbortSignal.timeout(45000)]),
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  const result = await response.json()
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : `Model load failed (HTTP ${response.status})`)
  return result
}

/** Format bytes as decimal GB to match the upstream checkpoint size estimates. */
const formatGigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`

const contextPresets = [
  ['8k', 8192], ['16k', 16384], ['32k', 32768], ['64k', 65536],
  ['128k', 131072], ['500k', 500000], ['1M', 1000000],
] as const

/**
 * Draft a context preset without persisting until the shared Load action.
 * Known architecture bounds disable oversized choices; an unknown bound is not
 * guessed. The parent owns the draft so model and context are submitted together.
 */
function ContextControl({ model, pending, value, change }: { model: ModelStatus; pending: boolean; value: number | null; change: (limit: number | null) => void }) {
  const customLimits = [...new Set([model.context_limit, value])].filter((limit): limit is number => limit !== null && !contextPresets.some(([, preset]) => preset === limit))
  return <div className="field model-context">
    <label htmlFor={`context-${model.id}`}>Max context length (tokens)</label>
    <select id={`context-${model.id}`} className="input" value={value?.toString() ?? ''} disabled={pending} onChange={event => change(event.target.value === '' ? null : Number(event.target.value))}>
      {contextPresets.map(([label, limit]) => <option key={limit} value={limit} disabled={model.architecture_context_limit !== null && limit > model.architecture_context_limit}>{label}</option>)}
      {customLimits.map(limit => <option key={limit} value={limit} disabled={model.architecture_context_limit !== null && limit > model.architecture_context_limit}>{limit} ({limit === model.context_limit ? 'saved' : 'draft'})</option>)}
      <option value="">Architecture maximum</option>
    </select>
  </div>
}

/** Match the native select chevron so the custom picker sits flush with other model rows. */
function Chevron() {
  return <svg className="model-picker-chevron" width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="m3 4.5 3 3 3-3" /></svg>
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

/**
 * An in-flow dropdown group with sibling choice/download controls, rather than
 * interactive children inside native options. Escape restores trigger focus;
 * arrows/Home/End move between enabled controls, Tab and outside clicks dismiss.
 */
function ModelPicker({ models, current, selectedId, pending, downloading, choose, download, pickerId = 'LLM', label = 'Language model' }: {
  pickerId?: string; label?: string; models: ModelStatus[]; current?: ModelStatus; selectedId: string | null; pending: boolean; downloading: boolean;
  choose: (model: ModelStatus) => void; download: (model: ModelStatus) => void;
}) {
  const [open, setOpen] = useState(false)
  const wrapper = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const popup = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    popup.current?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus()
    const outside = (event: PointerEvent) => { if (!wrapper.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [open])
  function dismiss() { setOpen(false); trigger.current?.focus() }
  return <div className="model-picker" ref={wrapper} onBlur={event => {
    if (!event.relatedTarget || !event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false)
  }} onKeyDown={event => {
    if (event.key === 'Escape' && open) { event.preventDefault(); event.stopPropagation(); dismiss() }
    if (!open || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const controls = Array.from(popup.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])
    const index = controls.indexOf(document.activeElement as HTMLButtonElement)
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? controls.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + controls.length) % controls.length
    controls[next]?.focus()
  }}>
    <button type="button" id={`model-${pickerId}`} className="input model-picker-trigger" ref={trigger} aria-expanded={open} aria-controls={`${pickerId}-model-picker`} disabled={!models.length} onClick={() => setOpen(value => !value)} onKeyDown={event => {
      if (!open && ['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); setOpen(true) }
    }}>
      <span>{current ? (current.display_name ?? current.repo_id.split('/')[1]) : `Choose a ${label.toLowerCase()}`}</span><Chevron />
    </button>
    {open && <div id={`${pickerId}-model-picker`} role="group" aria-label={`${label} options`} className="model-picker-options" ref={popup}>
      {models.map(model => <div className="model-picker-option" key={model.id}>
        <button type="button" className="model-picker-choice" disabled={pending || model.status !== 'complete'} aria-pressed={model.id === selectedId} onClick={() => { choose(model); dismiss() }}>
          <span>{(model.display_name ?? model.repo_id.split('/')[1])}</span>
          <small><span className="model-picker-size">{formatGigabytes(model.estimated_bytes)}</span><span className={`model-picker-status model-picker-status--${model.status}`}>{model.id === selectedId ? 'Selected' : model.status.replaceAll('_', ' ')}</span></small>
        </button>
        {model.status !== 'complete' && <button type="button" className="button model-download-icon" disabled={pending || downloading} aria-label={`${model.status === 'failed' || model.status === 'cancelled' ? 'Retry download' : 'Download'} ${(model.display_name ?? model.repo_id.split('/')[1])}`} title={`Download ${(model.display_name ?? model.repo_id.split('/')[1])}`} onClick={() => { download(model); dismiss() }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5" /></svg>
        </button>}
      </div>)}
    </div>}
  </div>
}

/**
 * Show live downloads and load a chosen model/context with one server request.
 * Polling and mutations have separate error state; request generations prevent
 * older polls from overwriting mutations. Unmount aborts outstanding requests.
 */
function LicenseNotice({ model, close, proceed }: { model: ModelStatus; close: () => void; proceed: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog className="model-license-dialog" ref={dialog} aria-labelledby="model-license-title" onCancel={event => { event.preventDefault(); close() }}>
    <h2 id="model-license-title">{model.display_name ?? model.repo_id} license</h2>
    <p>{model.license_notice} {model.license_url && <a href={model.license_url} target="_blank" rel="noreferrer">License</a>}</p>
    <div className="model-license-actions"><button type="button" className="button" autoFocus onClick={close}>Cancel</button><button type="button" className="button button--primary" onClick={proceed}>Continue</button></div>
  </dialog>
}

export default function SettingsPage() {
  const mutationVersion = useRef(0)
  const actionController = useRef<AbortController | null>(null)
  useEffect(() => () => { actionController.current?.abort() }, [])
  const [modelStatus, setModelStatus] = useState<ModelsResponse | null>(null)
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const [pending, setPending] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [imageId, setImageId] = useState<string | null>(null)
  const [videoId, setVideoId] = useState<string | null>(null)
  const [licenseModel, setLicenseModel] = useState<ModelStatus | null>(null)
  const licenseAction = useRef(false)
  const [chosenId, setChosenId] = useState<string | null>(null)
  const [contextDraft, setContextDraft] = useState<{ modelId: string; limit: number | null } | null>(null)
  const [lifecycle, setLifecycle] = useState<ModelLifecycleStatus | null>(null)
  const [loadPending, setLoadPending] = useState(false)
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
        const [next, runtime] = await Promise.all([requestModelStatus('', 'GET', undefined, controller.signal), requestLifecycle(controller.signal)])
        if (!controller.signal.aborted && !actionController.current && version === mutationVersion.current) { setModelStatus(next); setLifecycle(runtime); setError('') }
      } catch (failure) {
        if (!controller.signal.aborted && !actionController.current && version === mutationVersion.current) {
          setLifecycle(previous => previous?.state === 'loading' ? previous : null)
          setError(failure instanceof Error ? failure.message : 'Could not reach model service')
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1500)
      }
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [refresh])

  /**
   * Serialize a model mutation, update status on success, and retain its error
   * independently from polling. Version changes invalidate polls spanning it;
   * the controller also suppresses state updates after unmount cancellation.
   */
  async function submitModelChange(path: string, method: 'POST' | 'DELETE', body?: { license_acknowledged: boolean }) {
    if (actionController.current) return false
    const controller = new AbortController()
    actionController.current = controller
    mutationVersion.current += 1
    setPending(true)
    setActionError('')
    try {
      const next = await requestModelStatus(path, method, body, controller.signal)
      if (!controller.signal.aborted) { setModelStatus(next); setError(''); return true }
      return false
    } catch (failure) {
      if (!controller.signal.aborted) setActionError(failure instanceof Error ? failure.message : 'Request failed')
      return false
    } finally {
      mutationVersion.current += 1
      actionController.current = null
      if (!controller.signal.aborted) setPending(false)
    }
  }
  /** Require a fresh explicit acknowledgement for each licensed download or retry. */
  function downloadModel(model: ModelStatus) {
    if (actionController.current) return
    if (model.license_notice) { licenseAction.current = false; setLicenseModel(model) }
    else void submitModelChange(`/${model.id}/download`, 'POST')
  }
  /** Save selection/context and request actual loading as one guarded server action. */
  async function loadModel(model: ModelStatus, limit: number | null) {
    if (actionController.current) return
    const controller = new AbortController()
    actionController.current = controller
    mutationVersion.current += 1
    setPending(true)
    setLoadPending(true)
    setActionError('')
    try {
      const accepted = await requestLifecycle(controller.signal, { model_id: model.id, context_limit: limit })
      if (!controller.signal.aborted) setLifecycle(accepted)
      const [catalog, runtime] = await Promise.all([requestModelStatus('', 'GET', undefined, controller.signal), requestLifecycle(controller.signal)])
      if (!controller.signal.aborted) {
        setModelStatus(catalog)
        setLifecycle(runtime)
        if (catalog.selected_model_id === model.id) { setChosenId(null); setContextDraft(null) }
        setError('')
      }
    } catch (failure) {
      if (!controller.signal.aborted) setActionError(failure instanceof Error ? failure.message : 'Model load failed')
    } finally {
      mutationVersion.current += 1
      if (actionController.current === controller) actionController.current = null
      if (!controller.signal.aborted) { setPending(false); setLoadPending(false) }
    }
  }
  const models = modelStatus?.models ?? []
  const languageModels = models.filter(model => !model.kind || model.kind === 'llm')
  const imageModels = models.filter(model => model.kind === 'image')
  const image = imageModels.find(model => model.id === imageId) ?? imageModels[0]
  const videoModels = models.filter(model => model.kind === 'video')
  const video = videoModels.find(model => model.id === videoId) ?? videoModels[0]
  const downloading = models.some(model => model.status === 'downloading' || model.status === 'cancelling')
  const chosen = languageModels.find(model => model.id === (chosenId ?? modelStatus?.selected_model_id))
  const contextLimit = chosen && contextDraft?.modelId === chosen.id ? contextDraft.limit : chosen?.context_limit ?? null
  const validContext = contextLimit === null || (Number.isSafeInteger(contextLimit) && contextLimit > 0 && contextLimit <= 2147483647 && (chosen?.architecture_context_limit == null || contextLimit <= chosen.architecture_context_limit))
  const loading = loadPending || lifecycle?.state === 'loading'
  const busy = pending || loading || lifecycle?.state === 'unloading'
  const loaded = (lifecycle?.state === 'ready' || lifecycle?.state === 'offloaded') && lifecycle.model_id === chosen?.id && lifecycle.configured_context_limit === contextLimit
  const loadable = !busy && !!chosen && chosen.status === 'complete' && validContext && !loaded && !!lifecycle
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
    ?? (chosen && ['cancelled', 'failed'].includes(chosen.status) ? chosen : undefined)
  const imageDownload = imageModels.find(model => ['downloading', 'cancelling', 'failed', 'cancelled'].includes(model.status))
  const videoDownload = videoModels.find(model => ['downloading', 'cancelling', 'failed', 'cancelled'].includes(model.status))
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
              <button type="button" className={`button model-load ${loadable ? 'button--primary' : 'button--secondary'}`} disabled={!loadable} onClick={() => { if (chosen) void loadModel(chosen, contextLimit) }}>{loading ? 'Loading…' : loaded ? 'Loaded' : 'Load'}</button>
            </div>
            <div className="model-llm-fields">
              <div className="field">
                <span className="field-label" aria-hidden="true">Model</span>
                <ModelPicker models={languageModels} current={chosen} selectedId={chosen?.id ?? null} pending={busy} downloading={downloading}
                  choose={model => { setChosenId(model.id); setContextDraft(null) }}
                  download={model => { setChosenId(model.id); setContextDraft(null); downloadModel(model) }} />
              </div>
              {chosen
                ? <ContextControl model={chosen} pending={busy} value={contextLimit} change={limit => setContextDraft({ modelId: chosen.id, limit })} />
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
                    <button type="button" className="button" disabled={pending || model.status === 'cancelling' || (!active && downloading)} onClick={() => { setChosenId(model.id); if (active) void submitModelChange(`/${model.id}/download`, 'DELETE'); else downloadModel(model) }}>{active ? model.status === 'cancelling' ? 'Cancelling…' : 'Cancel download' : 'Retry download'}</button>
                  </div>
                  {active && <><progress className="progress" aria-label={`${model.id} download progress`} max={model.total_bytes || 1} value={model.total_bytes ? model.downloaded_bytes : undefined} /><span className="mono faint">{formatGigabytes(model.downloaded_bytes)} / {model.total_bytes ? formatGigabytes(model.total_bytes) : 'checking checkpoint size'}</span></>}
                  {model.error && <p role="alert" className="error">{model.error}</p>}
                </div>
              })}
            </div>
          </div>
          <div className="model-row model-row--video">
            <label htmlFor="model-Video"><ModelIcon type="Video" />Video</label>
            <div className="field"><ModelPicker pickerId="Video" label="Video model" models={videoModels} current={video} selectedId={videoId} pending={pending} downloading={downloading} choose={model => setVideoId(model.id)} download={model => { setVideoId(model.id); downloadModel(model) }} /></div>
            {video && !video.inference_available && <span className="muted model-video-note">Download only</span>}
            {videoDownload && <div className="model-download-status model-video-progress">
              <div className="model-download-heading"><p role="status">{videoDownload.display_name} · {videoDownload.status}</p>
                <button type="button" className="button" disabled={pending || videoDownload.status === 'cancelling' || (downloading && videoDownload.status !== 'downloading')} onClick={() => {
                  if (videoDownload.status === 'downloading') void submitModelChange(`/${videoDownload.id}/download`, 'DELETE')
                  else downloadModel(videoDownload)
                }}>{videoDownload.status === 'cancelling' ? 'Cancelling…' : videoDownload.status === 'downloading' ? 'Cancel download' : 'Retry download'}</button>
              </div>
              {['downloading', 'cancelling'].includes(videoDownload.status) && <><progress className="progress" aria-label={`${videoDownload.id} download progress`} max={videoDownload.total_bytes || 1} value={videoDownload.total_bytes ? videoDownload.downloaded_bytes : undefined} /><span className="mono faint">{formatGigabytes(videoDownload.downloaded_bytes)} / {videoDownload.total_bytes ? formatGigabytes(videoDownload.total_bytes) : 'checking checkpoint size'}</span></>}
              {videoDownload.error && <p role="alert" className="error">{videoDownload.error}</p>}
            </div>}
          </div>
          <div className="model-row model-row--video">
            <label htmlFor="model-Image"><ModelIcon type="Image" />Image</label>
            <div className="field"><ModelPicker pickerId="Image" label="Image model" models={imageModels} current={image} selectedId={imageId} pending={pending} downloading={downloading} choose={model => setImageId(model.id)} download={model => { setImageId(model.id); downloadModel(model) }} /></div>
            {image && !image.inference_available && <span className="muted model-video-note">Download only</span>}
            {imageDownload && <div className="model-download-status model-video-progress">
              <div className="model-download-heading"><p role="status">{imageDownload.display_name} · {imageDownload.status}</p>
                <button type="button" className="button" disabled={pending || imageDownload.status === 'cancelling' || (downloading && imageDownload.status !== 'downloading')} onClick={() => {
                  if (imageDownload.status === 'downloading') void submitModelChange(`/${imageDownload.id}/download`, 'DELETE')
                  else downloadModel(imageDownload)
                }}>{imageDownload.status === 'cancelling' ? 'Cancelling…' : imageDownload.status === 'downloading' ? 'Cancel download' : 'Retry download'}</button>
              </div>
              {['downloading', 'cancelling'].includes(imageDownload.status) && <><progress className="progress" aria-label={`${imageDownload.id} download progress`} max={imageDownload.total_bytes || 1} value={imageDownload.total_bytes ? imageDownload.downloaded_bytes : undefined} /><span className="mono faint">{formatGigabytes(imageDownload.downloaded_bytes)} / {imageDownload.total_bytes ? formatGigabytes(imageDownload.total_bytes) : 'checking checkpoint size'}</span></>}
              {imageDownload.error && <p role="alert" className="error">{imageDownload.error}</p>}
            </div>}
          </div>
          {modelSettings.filter(model => model.type !== 'LLM' && model.type !== 'Video' && model.type !== 'Image').map(model => <div className="model-row" key={model.type}>
            <label htmlFor={`model-${model.type}`}><ModelIcon type={model.type} />{model.label}</label>
            <select className="input" id={`model-${model.type}`} value={model.selected} disabled>
              {model.options.map(option => <option key={option}>{option}</option>)}
            </select>
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
