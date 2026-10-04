import { useEffect, useRef, useState } from 'react'
import { Panel, SectionHeading } from '../components/Controls'
import ModelLifecyclePanel from '../components/ModelLifecyclePanel'
import './SettingsPage.css'

type Model = {
  id: string
  repo_id: string
  revision: string
  license: string
  estimated_bytes: number
  status: 'not_downloaded' | 'downloading' | 'cancelling' | 'cancelled' | 'failed' | 'complete'
  downloaded_bytes: number
  total_bytes: number
  error: string | null
}
type Models = { models: Model[]; selected_model_id: string | null }

async function request(path = '', method = 'GET', body?: object, signal?: AbortSignal): Promise<Models> {
  const response = await fetch(`/v1/models${path}`, {
    method, signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!response.ok) {
    let message = `Model service returned HTTP ${response.status}`
    try {
      const data = await response.json()
      if (typeof data.detail === 'string') message = data.detail
    } catch { /* A proxy error may not be JSON. */ }
    throw new Error(message)
  }
  return response.json()
}
const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`

export default function SettingsPage() {
  const mutationVersion = useRef(0)
  const actionController = useRef<AbortController | null>(null)
  useEffect(() => () => { actionController.current?.abort() }, [])
  const [data, setData] = useState<Models | null>(null)
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const [pending, setPending] = useState(false)
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      const version = mutationVersion.current
      try {
        const next = await request('', 'GET', undefined, controller.signal)
        if (!controller.signal.aborted && !actionController.current && version === mutationVersion.current) { setData(next); setError('') }
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Could not reach model service')
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1500)
      }
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [refresh])

  async function act(path: string, method: string, body?: object) {
    if (actionController.current) return
    const controller = new AbortController()
    actionController.current = controller
    mutationVersion.current += 1
    setPending(true)
    setActionError('')
    try {
      const next = await request(path, method, body, controller.signal)
      if (!controller.signal.aborted) setData(next)
    } catch (failure) {
      if (!controller.signal.aborted) setActionError(failure instanceof Error ? failure.message : 'Request failed')
    } finally {
      mutationVersion.current += 1
      actionController.current = null
      if (!controller.signal.aborted) setPending(false)
    }
  }
  const downloading = data?.models.some(model => model.status === 'downloading' || model.status === 'cancelling')
  return (
    <div className="scroll-page">
      <div className="settings-layout stack">
        <ModelLifecyclePanel selectedModelId={data?.selected_model_id ?? null} />
        <SectionHeading>Language models</SectionHeading>
        <p>Download a checkpoint, then select it for loading. Downloads require the listed disk space plus a 1 GiB reserve. Selection does not load the model into memory.</p>
        <p>These checkpoints use Apache 2.0 or MIT licenses. Review each model card and its usage terms before downloading. Other modalities are not configured yet.</p>
        {error && <div role="alert">{error} <button type="button" onClick={() => setRefresh(value => value + 1)}>Refresh</button></div>}
        {actionError && <p role="alert">{actionError}</p>}
        {!data && !error && <p role="status">Loading model catalog…</p>}
        {data?.models.map(model => {
          const active = model.status === 'downloading' || model.status === 'cancelling'
          const selected = data.selected_model_id === model.id
          return <Panel className="checkpoint-card" key={model.id}>
            <h3>{model.id[0].toUpperCase() + model.id.slice(1)} — {model.repo_id.split('/')[1]}</h3>
            <a href={`https://huggingface.co/${model.repo_id}/tree/${model.revision}`} target="_blank" rel="noreferrer">Model card and pinned files</a>
            <p>{gb(model.estimated_bytes)} estimated · {model.license} · revision {model.revision.slice(0, 8)}</p>
            <p role="status">{model.status.replaceAll('_', ' ')}{selected ? ' · Selected' : ''}</p>
            {active && <>
              <progress aria-label={`${model.id} download progress`} max={model.total_bytes || 1} value={model.total_bytes ? model.downloaded_bytes : undefined} />
              <p>{gb(model.downloaded_bytes)} / {model.total_bytes ? gb(model.total_bytes) : 'checking checkpoint size'}</p>
            </>}
            {model.error && <p role="alert">{model.error}</p>}
            <div className="checkpoint-actions">
              {active ? <button type="button" disabled={pending || model.status === 'cancelling'} onClick={() => void act(`/${model.id}/download`, 'DELETE')}>{model.status === 'cancelling' ? 'Cancelling…' : 'Cancel download'}</button>
                : model.status !== 'complete' && <button type="button" disabled={pending || downloading} onClick={() => void act(`/${model.id}/download`, 'POST')}>{model.status === 'failed' || model.status === 'cancelled' ? 'Retry download' : 'Download'}</button>}
              <button type="button" disabled={pending || model.status !== 'complete' || selected} onClick={() => void act('/selection', 'PUT', { model_id: model.id })}>{selected ? 'Selected' : 'Select model'}</button>
            </div>
          </Panel>
        })}
      </div>
    </div>
  )
}
