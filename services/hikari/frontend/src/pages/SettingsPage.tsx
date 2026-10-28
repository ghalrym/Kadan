import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  chooseTier,
  downloadState,
  downloadExperimentalModel,
  fetchExperimentalModels,
  fetchSettings,
  licenseNotices,
  tierCanDownload,
  tierComplete,
  tierDownloading,
  tierProgress,
  type SettingsResponse,
  type CheckpointStatus,
  type TierModel,
  type TierSetting,
} from '../api/settings'
import { modalityLabels, tierCopy } from '../data/tiers'
import './SettingsPage.css'

/** Format bytes as decimal GB to match checkpoint size estimates. */
const gigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`

function downloadSize(model: CheckpointStatus): string {
  if (model.total_bytes > 0) return gigabytes(model.total_bytes)
  if (model.estimated_bytes > 0) return `${gigabytes(model.estimated_bytes)} (estimated)`
  return 'Size unknown'
}

const stateLabels = {
  unassigned: 'Not assigned yet',
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
      // Confirmation replaces the opener with the selected tier's progress button.
      document.getElementById(returnFocusId)?.focus()
    }
  }, [returnFocusId])
  return (
    <dialog
      className="tier-dialog"
      ref={dialog}
      aria-labelledby="tier-dialog-title"
      onCancel={(event) => {
        event.preventDefault()
        close()
      }}
    >
      <h2 id="tier-dialog-title">{title}</h2>
      {children}
    </dialog>
  )
}

function CheckIcon() {
  return (
    <svg className="tier-card-check" width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

/** A plain model name; hovering or focusing it reveals the checkpoint details. */
function ModelName({ id, model }: { id: string; model: NonNullable<TierModel['model']> }) {
  const tooltipId = `${id}-details`
  const details: [string, ReactNode][] = [
    ['Checkpoint', model.display_name ?? model.id],
    ['Repository', <span key="repo" className="mono">{model.repo_id}</span>],
    ['Download', downloadSize(model)],
  ]
  if (model.context_limit) details.push(['Context', `${model.context_limit.toLocaleString()} tokens`])
  if (model.license) details.push(['License', model.license])
  return (
    <span className="model-name" tabIndex={0} aria-describedby={tooltipId}>
      {model.short_name ?? model.display_name ?? model.id}
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

/** A pricing-style tile; the current tier, else the recommended one, is highlighted. */
function TierCard({ tier, current, highlighted, disabled, choose, showProgress }: {
  tier: TierSetting
  current: boolean
  highlighted: boolean
  disabled: boolean
  choose: () => void
  showProgress: () => void
}) {
  const copy = tierCopy[tier.id]
  const badge = current ? 'Current tier' : copy.recommended ? 'Recommended' : null
  const downloading = current && tierDownloading(tier)
  const ready = current && tierComplete(tier)
  return (
    <article
      className={`tier-card${highlighted ? ' tier-card--highlighted' : ''}${current ? ' tier-card--current' : ''}`}
      aria-current={current || undefined}
      aria-labelledby={`${tier.id}-name`}
    >
      {badge && <span className="tier-card-badge mono">{badge}</span>}
      <header className="tier-card-header">
        <h3 id={`${tier.id}-name`}>{tier.name}</h3>
        <p className="muted">{copy.tagline}</p>
      </header>
      <div className="tier-card-price">
        <p>
          <span className="tier-card-amount">{tier.vram_gb} GB</span>
          <span className="tier-card-unit">VRAM</span>
        </p>
        <p className="muted">with {tier.ram_gb} GB system RAM</p>
      </div>
      {current ? (
        <button
          id={`${tier.id}-action`}
          type="button"
          className="button tier-card-button"
          disabled={ready}
          onClick={showProgress}
        >
          {ready ? 'Your current tier' : downloading ? `Downloading… ${tierProgress(tier)}%` : 'View downloads'}
        </button>
      ) : (
        <button
          id={`${tier.id}-action`}
          type="button"
          className={`button tier-card-button${highlighted ? ' button--primary' : ''}`}
          disabled={disabled}
          onClick={choose}
        >
          Choose {tier.name}
        </button>
      )}
      <div className="tier-card-includes">
        <p className="tier-card-includes-label mono">Includes</p>
        <ul>
          {tier.models.map(({ modality, model }) => (
            <li key={modality}>
              <CheckIcon />
              <span>
                <span className="muted">{modalityLabels[modality].short}</span>{' '}
                {model ? <ModelName id={`${tier.id}-${modality}`} model={model} /> : 'TBD'}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </article>
  )
}

function DownloadProgress({ tier }: { tier: TierSetting }) {
  return (
    <ul className="tier-downloads">
      {tier.models.map(({ modality, model }) => {
        const state = downloadState(model)
        const label = modalityLabels[modality].label
        const size = model ? model.total_bytes || model.estimated_bytes : 0
        return (
          <li key={modality} className={`tier-download tier-download--${state}`}>
            <div className="tier-download-heading">
              <span>
                {label}
                {model && <span className="muted"> · {model.display_name ?? model.id}</span>}
              </span>
              <span className="mono tier-download-status">{stateLabels[state]}</span>
            </div>
            {model && (
              <>
                {/* A queued download is indeterminate until the worker reports its size. */}
                <progress
                  className="progress"
                  aria-label={`${label} download progress`}
                  max={size || 1}
                  value={
                    state === 'queued'
                      ? undefined
                      : state === 'complete'
                        ? size || 1
                        : model.downloaded_bytes
                  }
                />
                <span className="mono faint">
                  {state === 'complete' || !size
                    ? downloadSize(model)
                    : `${gigabytes(model.downloaded_bytes)} / ${downloadSize(model)}`}
                </span>
              </>
            )}
            {model?.error && <p role="alert" className="error">{model.error}</p>}
          </li>
        )
      })}
    </ul>
  )
}

/** Individual opt-in models are discovered from connected experimental workers. */
export function ExperimentalModelDownloads() {
  const [models, setModels] = useState<CheckpointStatus[]>([])
  const [confirmModel, setConfirmModel] = useState<CheckpointStatus | null>(null)
  const [acknowledged, setAcknowledged] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [needsRefresh, setNeedsRefresh] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const action = useRef<AbortController | null>(null)
  const version = useRef(0)
  useEffect(() => () => action.current?.abort(), [])

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      const readingVersion = version.current
      try {
        if (!action.current) {
          const next = await fetchExperimentalModels(controller.signal)
          if (!controller.signal.aborted && !action.current && readingVersion === version.current) {
            setModels(next)
            if (next.length) timer = setTimeout(poll, 1500)
          }
        } else timer = setTimeout(poll, 1500)
      } catch (failure) {
        if (!controller.signal.aborted && readingVersion === version.current) {
          setError(failure instanceof Error ? failure.message : 'Could not load experimental models.')
        }
      }
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [refresh])

  async function refreshModels() {
    if (action.current) return
    const controller = new AbortController()
    action.current = controller
    version.current++
    setSubmitting(true)
    try {
      const next = await fetchExperimentalModels(AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]))
      if (!controller.signal.aborted) {
        setModels(next)
        setError('')
        setNeedsRefresh(false)
        setConfirmModel(null)
        setAcknowledged(false)
        setRefresh((value) => value + 1)
      }
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Could not refresh models.')
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  async function download() {
    if (!confirmModel || !acknowledged || action.current || needsRefresh) return
    const current = models.find((model) => model.id === confirmModel.id)
    if (!current || ['complete', 'downloading', 'cancelling'].includes(current.status)) return
    const controller = new AbortController()
    action.current = controller
    version.current++
    setSubmitting(true)
    setError('')
    try {
      const next = await downloadExperimentalModel(confirmModel.id, acknowledged,
        AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]))
      if (!controller.signal.aborted) {
        setModels(next)
        setConfirmModel(null)
        setAcknowledged(false)
        setRefresh((value) => value + 1)
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        setError(failure instanceof Error ? failure.message : 'Could not start model download.')
        setNeedsRefresh(true)
      }
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  if (!models.length && !error) return null
  const confirmed = models.find((model) => model.id === confirmModel?.id)
  const confirmedUnavailable = !confirmed || ['complete', 'downloading', 'cancelling'].includes(confirmed.status)
  const errorPanel = error && <div role="alert" className="error-panel">
    <p>{error}</p>
    {needsRefresh && <p>The download may have started. Refresh models before trying again.</p>}
    <button className="button" type="button" disabled={submitting} onClick={() => void refreshModels()}>Refresh models</button>
  </div>
  return <section className="experimental-models stack" aria-labelledby="experimental-models-title">
    <h2 id="experimental-models-title">Experimental models</h2>
    {errorPanel}
    <ul className="tier-downloads">
      {models.map((model) => {
        const state = downloadState(model)
        const size = model.total_bytes || model.estimated_bytes
        const active = model.status === 'downloading' || model.status === 'cancelling'
        const name = model.display_name ?? model.id
        return <li key={model.id} className={`tier-download tier-download--${state}`}>
          <div className="tier-download-heading">
            <strong>{name}</strong>
            <span className="mono tier-download-status">{model.status === 'cancelling' ? 'Cancelling' : stateLabels[state]}</span>
          </div>
          <p>{model.license_notice || 'This download includes pinned model-specific Python code.'}</p>
          <div className="experimental-model-actions">
            {model.license_url && <a href={model.license_url} target="_blank" rel="noreferrer">{model.license === 'lfm1.0' ? 'LFM Open License v1.0' : model.license}</a>}
            <span className="mono muted">{size ? `${gigabytes(model.downloaded_bytes)} / ${downloadSize(model)}` : 'Size unknown'}</span>
            <button id={`${model.id}-download`} className="button" type="button"
              aria-label={`Download ${name}`} disabled={submitting || active || model.status === 'complete' || needsRefresh}
              onClick={() => { setAcknowledged(false); setConfirmModel(model) }}>
              {model.status === 'complete' ? 'Downloaded' : active ? 'Downloading…' : model.status === 'failed' || model.status === 'cancelled' ? 'Retry download' : 'Download'}
            </button>
          </div>
          {active && <progress className="progress" aria-label={`${name} download progress`} max={size || 1} value={model.total_bytes ? model.downloaded_bytes : undefined} />}
          {model.error && <p className="error" role="alert">{model.error}</p>}
        </li>
      })}
    </ul>
    {confirmModel && <Modal title={`Download ${confirmModel.display_name ?? confirmModel.id}?`}
      close={() => !submitting && setConfirmModel(null)} returnFocusId={`${confirmModel.id}-download`}>
      <p className="tier-dialog-notice">{confirmModel.license_notice || 'This download includes pinned model-specific Python code.'}{' '}
        {confirmModel.license_url && <a href={confirmModel.license_url} target="_blank" rel="noreferrer">LFM Open License v1.0</a>}
      </p>
      <label className="experimental-license-choice"><input type="checkbox" checked={acknowledged} disabled={submitting}
        onChange={(event) => setAcknowledged(event.target.checked)} />I have reviewed and accept this model’s license.</label>
      {errorPanel}
      <div className="tier-dialog-actions">
        <button className="button" type="button" autoFocus disabled={submitting} onClick={() => setConfirmModel(null)}>Cancel</button>
        <button className="button button--primary" type="button" disabled={!acknowledged || submitting || needsRefresh || confirmedUnavailable} onClick={() => void download()}>
          {submitting ? 'Starting…' : 'Accept and download'}
        </button>
      </div>
    </Modal>}
  </section>
}

/**
 * Choose a hardware tier, confirm its download, and follow per-model progress.
 * Settings poll only while the selected tier downloads or its progress is open.
 */
export default function SettingsPage() {
  const [settings, setSettings] = useState<SettingsResponse | null>(null)
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const [needsRefresh, setNeedsRefresh] = useState(false)
  const [confirmTier, setConfirmTier] = useState<TierSetting | null>(null)
  const [progressOpen, setProgressOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const action = useRef<AbortController | null>(null)
  const settingsVersion = useRef(0)
  useEffect(() => () => action.current?.abort(), [])

  const currentTier = settings?.tier ?? null
  const selected = settings?.tiers.find((tier) => tier.id === currentTier)
  const downloading = !!selected && tierDownloading(selected)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    // Poll serially; a pending tier change owns the next snapshot.
    async function poll() {
      const version = settingsVersion.current
      try {
        if (!action.current) {
          const next = await fetchSettings(controller.signal)
          if (!controller.signal.aborted && !action.current && version === settingsVersion.current) {
            setSettings(next)
            setError('')
          }
        }
      } catch (failure) {
        if (!controller.signal.aborted && !action.current && version === settingsVersion.current)
          setError(failure instanceof Error ? failure.message : 'Could not load settings')
      }
      if (!controller.signal.aborted && (downloading || progressOpen))
        timer = setTimeout(poll, 1500)
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [downloading, progressOpen, refresh])

  async function refreshSavedSettings() {
    if (action.current) return
    const controller = new AbortController()
    action.current = controller
    settingsVersion.current++
    setSubmitting(true)
    setError('')
    try {
      const next = await fetchSettings(
        AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]),
      )
      if (!controller.signal.aborted) {
        setSettings(next)
        setNeedsRefresh(false)
      }
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(failure instanceof Error ? failure.message : 'Could not load settings')
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  async function confirm(tier: TierSetting) {
    if (action.current || needsRefresh) return
    const controller = new AbortController()
    action.current = controller
    settingsVersion.current++
    setSubmitting(true)
    setError('')
    setActionError('')
    try {
      const next = await chooseTier(
        tier.id,
        AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]),
      )
      if (!controller.signal.aborted) {
        setSettings(next)
        setConfirmTier(null)
        setProgressOpen(true)
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        setActionError(failure instanceof Error ? failure.message : 'Could not change tier')
        setNeedsRefresh(true)
        setConfirmTier(null)
      }
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  const notices = confirmTier ? licenseNotices(confirmTier) : []
  const errorPanel = (actionError || error) && (
    <div role="alert" className="error-panel tier-alert">
      <div>
        {actionError && <p>{actionError}</p>}
        {needsRefresh && <p>The change may have been saved. Refresh saved state before trying again.</p>}
        {error && <p>{error}</p>}
      </div>
      {needsRefresh ? (
        <button type="button" className="button" disabled={submitting} onClick={() => void refreshSavedSettings()}>
          {submitting ? 'Refreshing…' : 'Refresh saved state'}
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
      {confirmTier && (
        <Modal
          title={`Download ${confirmTier.name} models?`}
          close={() => !submitting && setConfirmTier(null)}
          returnFocusId={`${confirmTier.id}-action`}
        >
          <p className="muted">
            {currentTier ? `Switching to ${confirmTier.name} downloads` : 'This downloads'} the models
            sized for {confirmTier.vram_gb} GB VRAM and {confirmTier.ram_gb} GB RAM.
          </p>
          {notices.map((model) => (
            <p key={model.id} className="tier-dialog-notice">
              <strong>{model.display_name ?? model.id}:</strong> {model.license_notice}{' '}
              {model.license_url && (
                <a href={model.license_url} target="_blank" rel="noreferrer">License</a>
              )}
            </p>
          ))}
          {errorPanel}
          <div className="tier-dialog-actions">
            <button type="button" className="button" autoFocus disabled={submitting} onClick={() => setConfirmTier(null)}>
              No
            </button>
            <button type="button" className="button button--primary" disabled={submitting || needsRefresh} onClick={() => void confirm(confirmTier)}>
              {submitting ? 'Starting…' : notices.length ? 'Accept and download' : 'Yes, download'}
            </button>
          </div>
        </Modal>
      )}
      {progressOpen && selected && (
        <Modal
          title={
            downloading
              ? `Downloading ${selected.name} models`
              : tierComplete(selected)
                ? `${selected.name} models downloaded`
                : `${selected.name} downloads`
          }
          close={() => setProgressOpen(false)}
          returnFocusId={`${selected.id}-action`}
        >
          <DownloadProgress tier={selected} />
          {errorPanel}
          <div className="tier-dialog-actions">
            {tierCanDownload(selected) && (
              <button type="button" className="button" disabled={submitting || needsRefresh} onClick={() => void confirm(selected)}>
                Resume downloads
              </button>
            )}
            <button
              type="button"
              className={`button${downloading ? '' : ' button--primary'}`}
              onClick={() => setProgressOpen(false)}
            >
              {downloading ? 'Continue in background' : 'Done'}
            </button>
          </div>
        </Modal>
      )}
      <div className="settings-layout tier-layout stack">
        <header className="tier-intro">
          <h1>Choose your tier</h1>
          <p className="muted">Pick the one that matches your GPU and RAM.</p>
        </header>
        {errorPanel}
        {!settings && !error && <p role="status" className="muted">Loading tiers…</p>}
        {settings && (
          <div className="tier-cards">
            {settings.tiers.map((tier) => (
              <TierCard
                key={tier.id}
                tier={tier}
                current={tier.id === currentTier}
                highlighted={currentTier ? tier.id === currentTier : tierCopy[tier.id].recommended}
                disabled={submitting || downloading || needsRefresh}
                choose={() => setConfirmTier(tier)}
                showProgress={() => setProgressOpen(true)}
              />
            ))}
          </div>
        )}
        <ExperimentalModelDownloads />
      </div>
    </div>
  )
}
