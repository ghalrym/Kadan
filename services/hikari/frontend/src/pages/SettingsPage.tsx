import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  cancelModelDownload,
  configureModelContext,
  configuredModels,
  downloadModel,
  downloadState,
  fetchSettings,
  type CheckpointStatus,
  type SettingsResponse,
} from '../api/settings'
import './SettingsPage.css'

const gigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`
const stateLabels = {
  unassigned: 'Worker unavailable',
  not_downloaded: 'Not downloaded',
  queued: 'Queued',
  downloading: 'Downloading',
  complete: 'Downloaded',
  failed: 'Failed',
}

function Modal({
  title,
  close,
  returnFocusId,
  children,
}: {
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
      className="model-dialog"
      ref={dialog}
      aria-labelledby="model-dialog-title"
      onCancel={(event) => {
        event.preventDefault()
        close()
      }}
    >
      <h2 id="model-dialog-title">{title}</h2>
      {children}
    </dialog>
  )
}

/** The fixed stack's individual checkpoints, with explicit download and context changes. */
export function ModelDownloads() {
  const [settings, setSettings] = useState<SettingsResponse | null>(null)
  const [confirmModel, setConfirmModel] = useState<CheckpointStatus | null>(
    null,
  )
  const [acknowledged, setAcknowledged] = useState(false)
  const [contextModel, setContextModel] = useState<CheckpointStatus | null>(
    null,
  )
  const [contextValue, setContextValue] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [readError, setReadError] = useState('')
  const [actionError, setActionError] = useState('')
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
        if (action.current) {
          timer = setTimeout(poll, 1500)
          return
        }
        const next = await fetchSettings(controller.signal)
        if (
          !controller.signal.aborted &&
          !action.current &&
          readingVersion === version.current
        ) {
          setSettings(next)
          setReadError('')
          if (
            next.models.some(
              (model) =>
                model.status === 'downloading' || model.status === 'cancelling',
            )
          )
            timer = setTimeout(poll, 1500)
        }
      } catch (failure) {
        if (!controller.signal.aborted && readingVersion === version.current)
          setReadError(
            failure instanceof Error
              ? failure.message
              : 'Could not load models.',
          )
      }
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [refresh])

  async function refreshModels() {
    if (action.current) return
    const controller = new AbortController()
    action.current = controller
    version.current++
    setSubmitting(true)
    try {
      const next = await fetchSettings(
        AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]),
      )
      if (!controller.signal.aborted) {
        setSettings(next)
        setReadError('')
        setActionError('')
        setNeedsRefresh(false)
        setConfirmModel(null)
        setContextModel(null)
        setAcknowledged(false)
        setRefresh((value) => value + 1)
      }
    } catch (failure) {
      if (!controller.signal.aborted)
        setReadError(
          failure instanceof Error
            ? failure.message
            : 'Could not refresh models.',
        )
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  async function download() {
    if (!confirmModel || !acknowledged || action.current || needsRefresh) return
    const current = settings?.models.find(
      (model) => model.id === confirmModel.id,
    )
    if (
      !current ||
      ['complete', 'downloading', 'cancelling'].includes(current.status)
    )
      return
    const controller = new AbortController()
    action.current = controller
    version.current++
    setSubmitting(true)
    setActionError('')
    try {
      const models = await downloadModel(
        confirmModel.id,
        acknowledged,
        AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]),
      )
      if (!controller.signal.aborted) {
        setSettings((saved) => saved && { ...saved, models })
        setConfirmModel(null)
        setAcknowledged(false)
        setRefresh((value) => value + 1)
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        setActionError(
          failure instanceof Error
            ? failure.message
            : 'Could not start model download.',
        )
        setNeedsRefresh(true)
      }
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  async function cancelDownload(model: CheckpointStatus) {
    if (action.current || needsRefresh) return
    const controller = new AbortController()
    action.current = controller
    version.current++
    setSubmitting(true)
    setActionError('')
    try {
      const models = await cancelModelDownload(model.id, controller.signal)
      if (!controller.signal.aborted) {
        setSettings((saved) => saved && { ...saved, models })
        setRefresh((value) => value + 1)
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        setActionError(
          failure instanceof Error
            ? failure.message
            : 'Could not cancel model download.',
        )
        setNeedsRefresh(true)
      }
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  async function saveContext() {
    if (!contextModel || action.current || needsRefresh) return
    const limit = contextValue.trim() === '' ? null : Number(contextValue)
    if (
      limit !== null &&
      (!Number.isSafeInteger(limit) ||
        limit < 1 ||
        (contextModel.architecture_context_limit !== null &&
          limit > contextModel.architecture_context_limit))
    ) {
      setActionError(
        'Enter a positive whole-token limit within the model maximum, or leave it blank to use the maximum.',
      )
      return
    }
    const controller = new AbortController()
    action.current = controller
    version.current++
    setSubmitting(true)
    setActionError('')
    try {
      const models = await configureModelContext(
        contextModel.id,
        limit,
        AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]),
      )
      if (!controller.signal.aborted) {
        setSettings((saved) => saved && { ...saved, models })
        setContextModel(null)
        setRefresh((value) => value + 1)
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        setActionError(
          failure instanceof Error
            ? failure.message
            : 'Could not save context.',
        )
        setNeedsRefresh(true)
      }
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  const errorPanel = (readError || actionError) && (
    <div role="alert" className="error-panel">
      {actionError && <p>{actionError}</p>}
      {readError && <p>{readError}</p>}
      {needsRefresh && (
        <p>
          The change may have been saved. Refresh models before trying again.
        </p>
      )}
      <button
        className="button"
        type="button"
        disabled={submitting}
        onClick={() => void refreshModels()}
      >
        Refresh models
      </button>
    </div>
  )
  const confirmed = settings?.models.find(
    (model) => model.id === confirmModel?.id,
  )
  const confirmedUnavailable =
    !confirmed ||
    ['complete', 'downloading', 'cancelling'].includes(confirmed.status)
  return (
    <section className="model-downloads stack" aria-labelledby="models-title">
      <header className="model-intro">
        <div>
          <h1 id="models-title">Models</h1>
          <p className="muted">
            One fixed stack. Review each model’s license before downloading.
          </p>
        </div>
        <button
          className="button"
          type="button"
          disabled={submitting}
          onClick={() => void refreshModels()}
        >
          Refresh models
        </button>
      </header>
      {errorPanel}
      {!settings && !readError && <p role="status">Loading models…</p>}
      <ul className="model-list">
        {settings &&
          configuredModels(settings).map(({ id, uses, checkpoint: model }) => {
            const name = model?.display_name ?? id
            const state = downloadState(model)
            const size = model ? model.total_bytes || model.estimated_bytes : 0
            const active =
              model?.status === 'downloading' || model?.status === 'cancelling'
            return (
              <li
                key={id}
                className={`model-download model-download--${model?.status === 'cancelled' ? 'cancelled' : state}`}
              >
                <div className="model-download-heading">
                  <strong>{name}</strong>
                  <span className="mono model-download-status">
                    {model?.status === 'cancelling'
                      ? 'Cancelling'
                      : model?.status === 'cancelled'
                        ? 'Cancelled'
                        : stateLabels[state]}
                  </span>
                </div>
                <p className="muted">{uses.join(' · ')}</p>
                {!model ? (
                  <p role="status">
                    Connect this model’s worker to check its download and
                    inference status.
                  </p>
                ) : (
                  <>
                    {model.license_notice && <p>{model.license_notice}</p>}
                    {!model.inference_available && (
                      <p role="status">
                        Inference is unavailable for this model.
                      </p>
                    )}
                    <div className="model-actions">
                      {model.license_url && (
                        <a
                          href={model.license_url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {model.license}
                        </a>
                      )}
                      <span className="mono muted">
                        {gigabytes(model.downloaded_bytes)} / {gigabytes(size)}
                      </span>
                      {active ? (
                        <button
                          id={`${id}-download`}
                          className="button"
                          type="button"
                          disabled={
                            submitting ||
                            needsRefresh ||
                            model.status === 'cancelling'
                          }
                          onClick={() => void cancelDownload(model)}
                        >
                          {model.status === 'cancelling'
                            ? 'Cancelling…'
                            : 'Cancel download'}
                        </button>
                      ) : (
                        <button
                          id={`${id}-download`}
                          className="button"
                          type="button"
                          aria-label={`Download ${name}`}
                          disabled={
                            submitting ||
                            needsRefresh ||
                            model.status === 'complete'
                          }
                          onClick={() => {
                            setActionError('')
                            setAcknowledged(false)
                            setConfirmModel(model)
                          }}
                        >
                          {model.status === 'complete'
                            ? 'Downloaded'
                            : state === 'failed'
                              ? 'Retry download'
                              : 'Download'}
                        </button>
                      )}
                      {model.kind === 'llm' && (
                        <button
                          id={`${id}-context`}
                          className="button"
                          type="button"
                          disabled={submitting || needsRefresh}
                          onClick={() => {
                            setActionError('')
                            setContextValue(
                              model.context_limit?.toString() ?? '',
                            )
                            setContextModel(model)
                          }}
                        >
                          Context
                        </button>
                      )}
                    </div>
                    {model.kind === 'llm' && (
                      <p className="faint">
                        Saved context:{' '}
                        {model.context_limit?.toLocaleString() ??
                          'model maximum'}{' '}
                        tokens
                        {model.architecture_context_limit
                          ? ` · Maximum ${model.architecture_context_limit.toLocaleString()}`
                          : ''}
                      </p>
                    )}
                    {active && (
                      <progress
                        className="progress"
                        aria-label={`${name} download progress`}
                        max={size || 1}
                        value={
                          model.total_bytes ? model.downloaded_bytes : undefined
                        }
                      />
                    )}
                    {model.error && (
                      <p className="error" role="alert">
                        {model.error}
                      </p>
                    )}
                  </>
                )}
              </li>
            )
          })}
      </ul>
      {confirmModel && (
        <Modal
          title={`Download ${confirmModel.display_name ?? confirmModel.id}?`}
          close={() => !submitting && setConfirmModel(null)}
          returnFocusId={`${confirmModel.id}-download`}
        >
          <p className="model-dialog-notice">
            {confirmModel.license_notice ||
              'Review this model’s license before downloading.'}{' '}
            {confirmModel.license_url && (
              <a
                href={confirmModel.license_url}
                target="_blank"
                rel="noreferrer"
              >
                {confirmModel.license}
              </a>
            )}
          </p>
          <label className="model-license-choice">
            <input
              type="checkbox"
              checked={acknowledged}
              disabled={submitting}
              onChange={(event) => setAcknowledged(event.target.checked)}
            />
            I have reviewed and accept this model’s license.
          </label>
          {errorPanel}
          <div className="model-dialog-actions">
            <button
              className="button"
              type="button"
              autoFocus
              disabled={submitting}
              onClick={() => setConfirmModel(null)}
            >
              Cancel
            </button>
            <button
              className="button button--primary"
              type="button"
              disabled={
                !acknowledged ||
                submitting ||
                needsRefresh ||
                confirmedUnavailable
              }
              onClick={() => void download()}
            >
              {submitting ? 'Starting…' : 'Accept and download'}
            </button>
          </div>
        </Modal>
      )}
      {contextModel && (
        <Modal
          title={`Context for ${contextModel.display_name ?? contextModel.id}`}
          close={() => !submitting && setContextModel(null)}
          returnFocusId={`${contextModel.id}-context`}
        >
          <label className="stack">
            Token limit
            <input
              type="number"
              min="1"
              max={contextModel.architecture_context_limit ?? undefined}
              value={contextValue}
              disabled={submitting}
              placeholder="Model maximum"
              onChange={(event) => setContextValue(event.target.value)}
            />
          </label>
          <p className="faint">
            Leave blank to use the model maximum. Larger contexts need more
            memory. This applies to later requests.
          </p>
          {errorPanel}
          <div className="model-dialog-actions">
            <button
              className="button"
              type="button"
              disabled={submitting}
              onClick={() => setContextModel(null)}
            >
              Cancel
            </button>
            <button
              className="button button--primary"
              type="button"
              disabled={submitting || needsRefresh}
              onClick={() => void saveContext()}
            >
              Save context
            </button>
          </div>
        </Modal>
      )}
    </section>
  )
}

export default function SettingsPage() {
  return (
    <div className="scroll-page">
      <div className="settings-layout stack">
        <ModelDownloads />
      </div>
    </div>
  )
}
