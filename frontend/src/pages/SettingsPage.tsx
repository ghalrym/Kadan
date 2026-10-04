import { useEffect, useRef, useState } from 'react'
import { Panel, SectionHeading } from '../components/Controls'
import ModelLifecyclePanel from '../components/ModelLifecyclePanel'
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
  return <div className="field">
    <label htmlFor={`context-${model.id}`}>Context window (tokens)</label>
    <input id={`context-${model.id}`} className="input" type="number" min={1} max={model.architecture_context_limit ?? 2147483647} step={1} value={value} disabled={pending} onChange={event => setValue(event.target.value)} placeholder="Architecture maximum" />
    <p>Saved: {model.context_limit ?? 'Architecture maximum'}. Architecture maximum: {model.architecture_context_limit ?? 'available after download'}. Blank uses the architecture maximum on load; larger windows require more memory. No silent reduction.</p>
    <button type="button" className="button button--secondary" disabled={pending || !valid || parsed === model.context_limit} onClick={() => save(parsed)}>Save context window</button>
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
    if (actionController.current) return
    const controller = new AbortController()
    actionController.current = controller
    mutationVersion.current += 1
    setPending(true)
    setActionError('')
    try {
      const next = await requestModelStatus(path, method, body, controller.signal)
      if (!controller.signal.aborted) { setModelStatus(next); setError('') }
    } catch (failure) {
      if (!controller.signal.aborted) setActionError(failure instanceof Error ? failure.message : 'Request failed')
    } finally {
      mutationVersion.current += 1
      actionController.current = null
      if (!controller.signal.aborted) setPending(false)
    }
  }
  const downloading = modelStatus?.models.some(model => model.status === 'downloading' || model.status === 'cancelling')
  return (
    <div className="scroll-page">
      <div className="settings-layout stack">
        <ModelLifecyclePanel selectedModelId={modelStatus?.selected_model_id ?? null} />
        <SectionHeading>Language models</SectionHeading>
        <p>Download a checkpoint, then select it for loading. Downloads require the listed disk space plus a 1 GiB reserve. Selection does not load the model into memory.</p>
        <p>These checkpoints use Apache 2.0 or MIT licenses. Review each model card and its usage terms before downloading. Other modalities are not configured yet.</p>
        {error && <div role="alert">{error} <button type="button" className="button button--secondary" onClick={() => setRefresh(value => value + 1)}>Refresh</button></div>}
        {actionError && <p role="alert">{actionError}</p>}
        {!modelStatus && !error && <p role="status">Loading model catalog…</p>}
        {modelStatus?.models.map(model => {
          const active = model.status === 'downloading' || model.status === 'cancelling'
          const selected = modelStatus.selected_model_id === model.id
          return <Panel className="checkpoint-card" key={model.id}>
            <h3>{model.id[0].toUpperCase() + model.id.slice(1)} — {model.repo_id.split('/')[1]}</h3>
            <a href={`https://huggingface.co/${model.repo_id}/tree/${model.revision}`} target="_blank" rel="noreferrer">Model card and pinned files</a>
            <p>{formatGigabytes(model.estimated_bytes)} estimated · {model.license} · revision {model.revision.slice(0, 8)}</p>
            <p role="status">{model.status.replaceAll('_', ' ')}{selected ? ' · Selected' : ''}</p>
            {active && <>
              <progress aria-label={`${model.id} download progress`} max={model.total_bytes || 1} value={model.total_bytes ? model.downloaded_bytes : undefined} />
              <p>{formatGigabytes(model.downloaded_bytes)} / {model.total_bytes ? formatGigabytes(model.total_bytes) : 'checking checkpoint size'}</p>
            </>}
            {model.error && <p role="alert">{model.error}</p>}
            <ContextControl key={`${model.id}-${model.context_limit}`} model={model} pending={pending} save={limit => void submitModelChange(`/${model.id}/context`, 'PUT', { context_limit: limit })} />
            <div className="checkpoint-actions">
              {active ? <button type="button" className="button button--secondary" disabled={pending || model.status === 'cancelling'} onClick={() => void submitModelChange(`/${model.id}/download`, 'DELETE')}>{model.status === 'cancelling' ? 'Cancelling…' : 'Cancel download'}</button>
                : model.status !== 'complete' && <button type="button" className="button button--primary" disabled={pending || downloading} onClick={() => void submitModelChange(`/${model.id}/download`, 'POST')}>{model.status === 'failed' || model.status === 'cancelled' ? 'Retry download' : 'Download'}</button>}
              <button type="button" className="button button--secondary" disabled={pending || model.status !== 'complete' || selected} onClick={() => void submitModelChange('/selection', 'PUT', { model_id: model.id })}>{selected ? 'Selected' : 'Select model'}</button>
            </div>
          </Panel>
        })}
      </div>
    </div>
  )
}
