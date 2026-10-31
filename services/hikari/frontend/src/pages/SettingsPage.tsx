import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  downloadModels,
  downloadState,
  fetchModels,
  licenseNotices,
  modelsCanDownload,
  modelsComplete,
  modelsDownloading,
  modelsProgress,
  type CapabilityModel,
  type CheckpointStatus,
} from '../api/settings'
import './SettingsPage.css'

/** Format bytes as decimal GB to match checkpoint size estimates. */
const gigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`

function downloadSize(model: CheckpointStatus): string {
  if (model.total_bytes > 0) return gigabytes(model.total_bytes)
  if (model.estimated_bytes > 0) return `${gigabytes(model.estimated_bytes)} (estimated)`
  return 'Size unknown'
}

/** The catalog still marks the LFM checkpoints experimental; here they are the defaults. */
const modelName = (model: CheckpointStatus) =>
  (model.display_name ?? model.id).replace(/ \(experimental\)$/, '')

const stateLabels = {
  unavailable: 'Unavailable',
  not_downloaded: 'Not downloaded',
  queued: 'Queued',
  downloading: 'Downloading',
  complete: 'Complete',
  failed: 'Failed',
}

/** Open a native modal on mount; Escape runs `close` instead of dismissing directly. */
function Modal({ title, close, returnFocusId, children }: {
  title: string
  close: () => void
  returnFocusId: string
  children: ReactNode
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const element = dialog.current
    element?.showModal()
    return () => {
      element?.close()
      document.getElementById(returnFocusId)?.focus()
    }
  }, [returnFocusId])
  return (
    <dialog
      className="models-dialog"
      ref={dialog}
      aria-labelledby="models-dialog-title"
      onCancel={(event) => {
        event.preventDefault()
        close()
      }}
    >
      <h2 id="models-dialog-title">{title}</h2>
      {children}
    </dialog>
  )
}

/** A plain model name; hovering or focusing it reveals the checkpoint details. */
function ModelName({ id, model }: { id: string; model: CheckpointStatus }) {
  const tooltipId = `${id}-details`
  const details: [string, ReactNode][] = [
    ['Checkpoint', modelName(model)],
    ['Repository', <span key="repo" className="mono">{model.repo_id}</span>],
    ['Download', downloadSize(model)],
  ]
  if (model.context_limit) details.push(['Context', `${model.context_limit.toLocaleString()} tokens`])
  if (model.license) details.push(['License', model.license])
  return (
    <span className="model-name" tabIndex={0} aria-describedby={tooltipId}>
      {modelName(model)}
      <span role="tooltip" id={tooltipId} className="model-tooltip">
        <dl>
          {details.map(([term, value]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </span>
    </span>
  )
}

/** Bytes a checkpoint occupies once complete, using the estimate until the worker reports. */
const modelSize = (model: CheckpointStatus | null) => (model ? model.total_bytes || model.estimated_bytes : 0)

/** Share of a checkpoint already on disk, in percent. */
function modelFill(model: CheckpointStatus | null): number {
  const size = modelSize(model)
  if (!model || !size) return 0
  if (model.status === 'complete') return 100
  return Math.min(100, (model.downloaded_bytes / size) * 100)
}

/** Disk use split by capability; each segment fills in as its model downloads. */
function StorageBar({ rows }: { rows: CapabilityModel[] }) {
  return (
    <div className="models-storage" aria-hidden="true">
      {rows.map(({ capability, model }) => (
        <span
          key={capability.id}
          className={`models-storage-segment tone-${capability.tone}`}
          style={{ flexGrow: Math.max(modelSize(model), 1) }}
          title={capability.label}
        >
          <span style={{ width: `${modelFill(model)}%` }} />
        </span>
      ))}
    </div>
  )
}

function ModelRow({ capability, model, selected, disabled, onSelect }: CapabilityModel & {
  selected: boolean
  disabled: boolean
  onSelect: (selected: boolean) => void
}) {
  const state = downloadState(model)
  const size = modelSize(model)
  const active = state === 'queued' || state === 'downloading'
  return (
    <li className={`model-row model-row--${state} tone-${capability.tone}`}>
      <label className="model-row-choice">
        <input
          type="checkbox"
          aria-label={`Select ${capability.label} for download`}
          checked={selected}
          disabled={disabled}
          onChange={(event) => onSelect(event.target.checked)}
        />
      </label>
      <span className="model-row-swatch" aria-hidden="true" />
      <span className="model-row-label">
        <span className="model-row-capability">{capability.label}</span>
        <span className="model-row-name">
          {model ? <ModelName id={capability.id} model={model} /> : <span className="faint">{capability.id}</span>}
        </span>
      </span>
      <span className="mono faint model-row-size">
        {model && (active && size ? `${gigabytes(model.downloaded_bytes)} / ${downloadSize(model)}` : downloadSize(model))}
      </span>
      <span className="mono model-row-status">
        <span className="status-dot" aria-hidden="true" />
        {stateLabels[state]}
      </span>
      {/* A queued download is indeterminate until the worker reports its size. */}
      {model && active && (
        <progress
          className="progress model-row-progress"
          aria-label={`${capability.label} download progress`}
          max={size || 1}
          value={state === 'queued' ? undefined : model.downloaded_bytes}
        />
      )}
      {model?.error && <p role="alert" className="error model-row-error">{model.error}</p>}
    </li>
  )
}

/**
 * Show Hikari's fixed model for each capability, confirm licenses once, and
 * follow download progress. Models poll only while a download is active.
 */
export default function SettingsPage() {
  const [rows, setRows] = useState<CapabilityModel[] | null>(null)
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const [needsRefresh, setNeedsRefresh] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [acknowledged, setAcknowledged] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const action = useRef<AbortController | null>(null)
  const version = useRef(0)
  useEffect(() => () => action.current?.abort(), [])

  const downloading = !!rows && modelsDownloading(rows)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    // Poll serially; a pending download request owns the next snapshot.
    async function poll() {
      const reading = version.current
      try {
        if (!action.current) {
          const next = await fetchModels(controller.signal)
          if (!controller.signal.aborted && !action.current && reading === version.current) {
            setRows(next)
            setError('')
          }
        }
      } catch (failure) {
        if (!controller.signal.aborted && !action.current && reading === version.current)
          setError(failure instanceof Error ? failure.message : 'Could not load models')
      }
      if (!controller.signal.aborted && downloading) timer = setTimeout(poll, 1500)
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [downloading, refresh])

  async function run(work: (signal: AbortSignal) => Promise<CapabilityModel[]>, failed: string) {
    if (action.current) return
    const controller = new AbortController()
    action.current = controller
    version.current++
    setSubmitting(true)
    setError('')
    setActionError('')
    try {
      const next = await work(AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]))
      if (!controller.signal.aborted) {
        setRows(next)
        setSelectedIds([])
        setNeedsRefresh(false)
        setConfirming(false)
        setRefresh((value) => value + 1)
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        setActionError(failure instanceof Error ? failure.message : failed)
        setNeedsRefresh(true)
      }
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  const complete = !!rows && modelsComplete(rows)
  const selected = rows?.filter((row) => selectedIds.includes(row.capability.id) && modelsCanDownload([row])) ?? []
  const notices = licenseNotices(selected)
  const selectedBytes = selected.reduce((total, { model }) => total + modelSize(model), 0)
  const downloaded = rows?.filter(({ model }) => model?.status === 'complete').length ?? 0

  const errorPanel = (actionError || error) && (
    <div role="alert" className="error-panel models-alert">
      <div>
        {actionError && <p>{actionError}</p>}
        {needsRefresh && <p>Some downloads may have started. Refresh before trying again.</p>}
        {error && <p>{error}</p>}
      </div>
      {needsRefresh ? (
        <button type="button" className="button" disabled={submitting} onClick={() => void run(fetchModels, 'Could not load models')}>
          {submitting ? 'Refreshing…' : 'Refresh'}
        </button>
      ) : error && (
        <button type="button" className="button" disabled={submitting} onClick={() => setRefresh((value) => value + 1)}>
          Retry
        </button>
      )}
    </div>
  )

  return (
    <div className="scroll-page">
      {confirming && rows && (
        <Modal
          title="Download selected models?"
          close={() => !submitting && setConfirming(false)}
          returnFocusId="models-download"
        >
          <p className="muted">
            This downloads {selected.length} {selected.length === 1 ? 'model' : 'models'}, about {gigabytes(selectedBytes)}.
          </p>
          {notices.map((model) => (
            <p key={model.id} className="models-dialog-notice">
              <strong>{modelName(model)}:</strong> {model.license_notice}{' '}
              {model.license_url && (
                <a href={model.license_url} target="_blank" rel="noreferrer">License</a>
              )}
            </p>
          ))}
          {notices.length > 0 && (
            <label className="models-license-choice">
              <input type="checkbox" checked={acknowledged} disabled={submitting}
                onChange={(event) => setAcknowledged(event.target.checked)} />
              I have reviewed and accept these licenses.
            </label>
          )}
          {errorPanel}
          <div className="models-dialog-actions">
            <button type="button" className="button" autoFocus disabled={submitting} onClick={() => setConfirming(false)}>
              Cancel
            </button>
            <button
              type="button"
              className="button button--primary"
              disabled={submitting || needsRefresh || selected.length === 0 || (notices.length > 0 && !acknowledged)}
              onClick={() => void run((signal) => downloadModels(selected, true, signal), 'Could not start downloads')}
            >
              {submitting ? 'Starting…' : notices.length ? 'Accept and download' : 'Download'}
            </button>
          </div>
        </Modal>
      )}
      <div className="settings-layout models-layout stack">
        <header className="models-intro">
          <h1>Models</h1>
        </header>
        {!confirming && errorPanel}
        {!rows && !error && <p role="status" className="muted">Loading models…</p>}
        {rows && (
          <section className="panel models-panel" aria-labelledby="models-summary-title">
            <header className="models-summary">
              <div className="models-summary-text">
                <h2 id="models-summary-title">
                  {complete ? 'All models downloaded' : `${downloaded} of ${rows.length} downloaded`}
                </h2>
                <p className="mono muted">
                  {downloading
                    ? `Downloading… ${modelsProgress(rows)}%`
                    : `${gigabytes(rows.reduce((total, { model }) => total + modelSize(model), 0))} total`}
                </p>
              </div>
              {!complete && (
                <button
                  id="models-download"
                  type="button"
                  className="button button--primary"
                  disabled={submitting || needsRefresh || selected.length === 0}
                  onClick={() => { setAcknowledged(false); setConfirming(true) }}
                >
                  {submitting ? 'Starting…' : selected.length ? `Download selected (${selected.length})` : 'Download selected'}
                </button>
              )}
            </header>
            <StorageBar rows={rows} />
            <ol className="model-list">
              {rows.map((row) => (
                <ModelRow
                  key={row.capability.id}
                  {...row}
                  selected={selected.some((item) => item.capability.id === row.capability.id)}
                  disabled={submitting || confirming || needsRefresh || !modelsCanDownload([row])}
                  onSelect={(checked) => setSelectedIds((ids) => checked
                    ? [...ids, row.capability.id]
                    : ids.filter((id) => id !== row.capability.id))}
                />
              ))}
            </ol>
          </section>
        )}
      </div>
    </div>
  )
}
