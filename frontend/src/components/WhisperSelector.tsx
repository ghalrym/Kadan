import { useEffect, useRef, useState } from 'react'
import type { ModelStatus } from '../api/generated'
import { ModelPicker } from './ModelPicker'

/** Persist the native Whisper choice without loading or downloading weights. */
export function WhisperSelector({ models, pending: downloadPending, downloading, download, cancel }: { models: ModelStatus[]; pending: boolean; downloading: boolean; download: (model: ModelStatus) => void; cancel: (model: ModelStatus) => void }) {
  const [catalog, setCatalog] = useState<{ models: string[]; selected: string | null } | null>(null)
  const [error, setError] = useState('')
  const [pending, setPending] = useState(false)
  const inFlight = useRef(false)
  useEffect(() => {
    const controller = new AbortController()
    fetch('/v1/audio/transcriptions/models', { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error('Cannot load Whisper models.')
      setCatalog(await response.json())
    }).catch((reason: Error) => { if (!controller.signal.aborted) setError(reason.message) })
    return () => controller.abort()
  }, [])
  async function select(model: string) {
    if (inFlight.current) return
    inFlight.current = true
    setPending(true)
    setError('')
    try {
      const response = await fetch('/v1/audio/transcriptions/models', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }), signal: AbortSignal.timeout(15000) })
      if (!response.ok) throw new Error('Cannot save Whisper selection.')
      setCatalog(await response.json())
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Cannot save Whisper selection.') }
    finally { inFlight.current = false; setPending(false) }
  }
  const current = models.find(model => model.id === `whisper-${catalog?.selected}`)
  const job = models.find(model => ['downloading', 'cancelling', 'failed', 'cancelled'].includes(model.status))
  const active = job?.status === 'downloading' || job?.status === 'cancelling'
  return <div className="stack compact"><ModelPicker pickerId="STT" label="Whisper model" models={models} current={current} selectedId={current?.id ?? null} pending={pending || downloadPending} downloading={downloading} choose={model => void select(model.id.slice('whisper-'.length))} download={download} />
    {job && <div className="model-download-status">
      <div className="model-download-heading"><p role="status">{job.display_name} · {job.status}</p><button type="button" className="button" disabled={downloadPending || job.status === 'cancelling' || (!active && downloading)} onClick={() => active ? cancel(job) : download(job)}>{active ? 'Cancel download' : 'Retry download'}</button></div>
      {active && <progress className="progress" aria-label="Whisper download progress" value={job.total_bytes ? job.downloaded_bytes : undefined} max={job.total_bytes || 1} />}
      {job.error && <p role="alert" className="error">{job.error}</p>}
    </div>}
    {error && <p role="alert" className="error">{error}</p>}
  </div>
}
