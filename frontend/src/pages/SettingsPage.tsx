import { useEffect, useRef, useState } from 'react'
import { Panel, SectionHeading } from '../components/Controls'
import ModelLifecyclePanel from '../components/ModelLifecyclePanel'
import { modelSettings } from '../data/playground'
import type { ContextRequest, ModelsResponse, ModelStatus } from '../api/generated'
import './SettingsPage.css'

/**
 * Request catalog status or a mutation under /v1/models with a 15-second timeout.
 * Combines caller cancellation with the timeout; throws backend detail when
 * available, or an HTTP error for non-JSON proxy failures. Returns typed status.
 */
async function requestModelStatus(path = '', method: 'GET' | 'PUT' | 'POST' | 'DELETE' = 'GET', body?: ContextRequest | { model_id: string }, signal?: AbortSignal): Promise<ModelsResponse> {
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
/** Format bytes as decimal GB to match the upstream checkpoint size estimates. */
const formatGigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`

/**
 * Edit one model's saved context limit. Blank means architecture maximum;
 * validation disables saving invalid values, while save owns persistence/errors.
 * The parent keys this control by the saved limit to reset its local draft.
 */
function ContextControl({ model, pending, save }: { model: ModelStatus; pending: boolean; save: (limit: number | null) => void }) {
  const [value, setValue] = useState(model.context_limit?.toString() ?? '')
  const parsed = value.trim() === '' ? null : Number(value)
  const valid = parsed === null || (Number.isSafeInteger(parsed) && parsed > 0 && parsed <= 2147483647 && (model.architecture_context_limit === null || parsed <= model.architecture_context_limit))
  return <div className="field model-context">
    <label htmlFor={`context-${model.id}`}>Max context length (tokens)</label>
    <input id={`context-${model.id}`} className="input" type="number" min={1} max={model.architecture_context_limit ?? 2147483647} step={1} value={value} disabled={pending} onChange={event => setValue(event.target.value)} placeholder="Architecture maximum" />
    <button type="button" className="button button--secondary" disabled={pending || !valid || parsed === model.context_limit} onClick={() => save(parsed)}>Save context window</button>
  </div>
}

/** Preserve the original modality icons alongside the compact settings rows. */
function ModelIcon({ type }: { type: string }) {
  return <svg className={`model-icon type-${type}`} width="13" height="13" viewBox="0 0 12 12" aria-hidden="true">
    <circle cx="6" cy="3" r="2.5" /><circle cx="8.85" cy="5.07" r="2.5" />
    <circle cx="7.76" cy="8.43" r="2.5" /><circle cx="4.24" cy="8.43" r="2.5" />
    <circle cx="3.15" cy="5.07" r="2.5" /><circle className="model-icon-center" cx="6" cy="6" r="1.5" />
  </svg>
}

/**
 * A nonmodal picker with sibling selection/download controls, rather than
 * interactive children inside native options. Escape restores trigger focus;
 * arrows/Home/End move between enabled controls, Tab and outside clicks dismiss.
 */
function ModelPicker({ models, current, selectedId, pending, downloading, choose, download }: {
  models: ModelStatus[]; current?: ModelStatus; selectedId: string | null; pending: boolean; downloading: boolean;
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
    <button type="button" id="model-LLM" className="input model-picker-trigger" ref={trigger} aria-haspopup="dialog" aria-expanded={open} aria-controls="language-model-picker" disabled={!models.length} onClick={() => setOpen(value => !value)} onKeyDown={event => {
      if (!open && ['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); setOpen(true) }
    }}>
      <span>{current?.repo_id.split('/')[1] ?? 'Choose a language model'}</span><span aria-hidden="true">▾</span>
    </button>
    {open && <div id="language-model-picker" role="dialog" aria-label="Language model options" className="model-picker-options" ref={popup}>
      {models.map(model => <div className="model-picker-option" key={model.id}>
        <button type="button" className="model-picker-choice" disabled={pending || model.status !== 'complete'} aria-pressed={model.id === selectedId} onClick={() => { choose(model); dismiss() }}>
          <span>{model.repo_id.split('/')[1]}</span><small>{model.id === selectedId ? 'Selected' : model.status.replaceAll('_', ' ')}</small>
        </button>
        {model.status !== 'complete' && <button type="button" className="button model-download-icon" disabled={pending || downloading} aria-label={`${model.status === 'failed' || model.status === 'cancelled' ? 'Retry download' : 'Download'} ${model.repo_id.split('/')[1]}`} title={`Download ${model.repo_id.split('/')[1]}`} onClick={() => { download(model); dismiss() }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5" /></svg>
        </button>}
      </div>)}
      <button type="button" className="button button--text" onClick={dismiss}>Close model options</button>
    </div>}
  </div>
}

/**
 * Show live checkpoint downloads and persist selection/context changes.
 * Polling and mutations have separate error state; request generations prevent
 * older polls from overwriting mutations. Unmount aborts outstanding requests.
 */
export default function SettingsPage() {
  const mutationVersion = useRef(0)
  const actionController = useRef<AbortController | null>(null)
  useEffect(() => () => { actionController.current?.abort() }, [])
  const [modelStatus, setModelStatus] = useState<ModelsResponse | null>(null)
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const [pending, setPending] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [chosenId, setChosenId] = useState<string | null>(null)
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
        const next = await requestModelStatus('', 'GET', undefined, controller.signal)
        if (!controller.signal.aborted && !actionController.current && version === mutationVersion.current) { setModelStatus(next); setError('') }
      } catch (failure) {
        if (!controller.signal.aborted && !actionController.current && version === mutationVersion.current) {
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
  async function submitModelChange(path: string, method: 'PUT' | 'POST' | 'DELETE', body?: ContextRequest | { model_id: string }) {
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
  const models = modelStatus?.models ?? []
  const downloading = models.some(model => model.status === 'downloading' || model.status === 'cancelling')
  const chosen = models.find(model => model.id === (chosenId ?? modelStatus?.selected_model_id))
  const downloadStatus = models.find(model => model.status === 'downloading' || model.status === 'cancelling')
    ?? (chosen && ['cancelled', 'failed'].includes(chosen.status) ? chosen : undefined)
  return (
    <div className="scroll-page">
      <div className="settings-layout stack">
        <ModelLifecyclePanel selectedModelId={modelStatus?.selected_model_id ?? null} />
        <SectionHeading>Models</SectionHeading>
        <Panel className="model-settings">
          <div className="model-row model-row--llm">
            <label htmlFor="model-LLM"><ModelIcon type="LLM" />Language model</label>
            <ModelPicker models={models} current={chosen} selectedId={modelStatus?.selected_model_id ?? null} pending={pending} downloading={downloading}
              choose={model => { void submitModelChange('/selection', 'PUT', { model_id: model.id }).then(saved => { if (saved) setChosenId(null) }) }}
              download={model => { setChosenId(model.id); void submitModelChange(`/${model.id}/download`, 'POST') }} />
            <div className="model-llm-details stack">
              {!modelStatus && !error && <p role="status">Loading model catalog…</p>}
              {error && <div role="alert">{error} <button type="button" className="button" onClick={() => setRefresh(value => value + 1)}>Refresh</button></div>}
              {actionError && <p role="alert">{actionError}</p>}
              {!chosen && <div className="field"><label htmlFor="context-unselected">Max context length (tokens)</label><input id="context-unselected" className="input" placeholder="Choose a language model" disabled /></div>}
              {chosen && <>
                <ContextControl key={`${chosen.id}-${chosen.context_limit}`} model={chosen} pending={pending} save={limit => void submitModelChange(`/${chosen.id}/context`, 'PUT', { context_limit: limit })} />
              </>}
              {(downloadStatus ? [downloadStatus] : []).map(model => {
                const active = model.status === 'downloading' || model.status === 'cancelling'
                return <div className="model-download-status stack compact" key={model.id}>
                  <p role="status">{model.repo_id.split('/')[1]} · {model.status}</p>
                  {active && <><progress aria-label={`${model.id} download progress`} max={model.total_bytes || 1} value={model.total_bytes ? model.downloaded_bytes : undefined} /><span className="faint">{formatGigabytes(model.downloaded_bytes)} / {model.total_bytes ? formatGigabytes(model.total_bytes) : 'checking checkpoint size'}</span></>}
                  {model.error && <p role="alert">{model.error}</p>}
                  <button type="button" className="button" disabled={pending || model.status === 'cancelling' || (!active && downloading)} onClick={() => { setChosenId(model.id); void submitModelChange(`/${model.id}/download`, active ? 'DELETE' : 'POST') }}>{active ? model.status === 'cancelling' ? 'Cancelling…' : 'Cancel download' : 'Retry download'}</button>
                </div>
              })}
            </div>
          </div>
          {modelSettings.filter(model => model.type !== 'LLM').map(model => <div className="model-row" key={model.type}>
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
