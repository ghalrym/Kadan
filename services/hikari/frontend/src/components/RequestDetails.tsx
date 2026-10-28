import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router'
import { ActionButton } from './Controls'
import type { RequestResponse } from '../api/generated/types.gen'
import { requestStatus } from '../api/requestStatus'
import { usePolling } from '../hooks/usePolling'

/** Retain the original drawer structure around actual observed summaries. */
export default function RequestDetails() {
  const { requestId } = useParams()
  const { data, error, refresh } = usePolling<RequestResponse>(
    `/v1/requests/${encodeURIComponent(requestId ?? '')}`,
  )
  const request = data?.request
  const [copyFeedback, setCopyFeedback] = useState<{
    requestId: string | undefined
    recordId: string
    message: string
    failed: boolean
    signal: AbortSignal
  } | null>(null)
  const copyAttempt = useRef<AbortController | null>(null)
  useEffect(() => () => copyAttempt.current?.abort(), [requestId, request?.id])

  async function copy(kind: 'details' | 'id') {
    if (!request) return
    copyAttempt.current?.abort()
    const attempt = new AbortController()
    copyAttempt.current = attempt
    setCopyFeedback(null)
    try {
      await navigator.clipboard.writeText(
        kind === 'details' ? JSON.stringify(request, null, 2) : request.id,
      )
      if (!attempt.signal.aborted)
        setCopyFeedback({
          requestId,
          recordId: request.id,
          message: kind === 'details' ? 'Details copied' : 'ID copied',
          failed: false,
          signal: attempt.signal,
        })
    } catch {
      if (!attempt.signal.aborted)
        setCopyFeedback({
          requestId,
          recordId: request.id,
          message:
            kind === 'details'
              ? 'Could not copy details. Select the JSON under Advanced and copy it manually.'
              : 'Could not copy request ID.',
          failed: true,
          signal: attempt.signal,
        })
    }
  }
  const feedback =
    copyFeedback?.requestId === requestId &&
    copyFeedback?.recordId === request?.id &&
    !copyFeedback?.signal.aborted
      ? copyFeedback
      : null
  return (
    <div className="detail-overlay">
      <Link
        to="/requests"
        className="detail-backdrop"
        aria-label="Close request details"
        tabIndex={-1}
      />
      <aside className="request-drawer" aria-label="Request details">
        {request ? (
          <>
            <header className="drawer-header">
              <div className="row wrap">
                <span className={`badge type-${request.type}`}>
                  {request.type}
                </span>
                <span
                  className={`badge status-badge ${requestStatus(request).className}`}
                >
                  {requestStatus(request).label}
                </span>
                <Link to="/requests" className="button push-right">
                  Close
                </Link>
              </div>
              <p className="mono endpoint">
                <span className="muted">POST</span> {request.endpoint}
              </p>
            </header>
            <div className="drawer-content stack">
              <div className="detail-metrics mono muted">
                <span>
                  Model <strong>{request.model ?? '—'}</strong>
                </span>
                <span>
                  HTTP latency <strong>{request.latency}</strong>
                </span>
              </div>
              <section className="stack compact">
                <h3 className="eyebrow">Request summary</h3>
                <div className="detail-text">{request.prompt}</div>
              </section>
              {request.status < 400 ? (
                <section className="stack compact">
                  <h3 className="eyebrow">HTTP response summary</h3>
                  <div className="detail-text">{request.output}</div>
                </section>
              ) : (
                <div className="error-panel stack compact">
                  <h3>HTTP {request.status}</h3>
                  <p>{request.output}</p>
                </div>
              )}
              <details className="advanced-details">
                <summary>Advanced</summary>
                <div className="stack">
                  <h3 className="eyebrow">Observed request</h3>
                  <pre>{JSON.stringify(request, null, 2)}</pre>
                </div>
              </details>
              {feedback && (
                <p role={feedback.failed ? 'alert' : 'status'}>
                  {feedback.message}
                </p>
              )}
            </div>
            <footer className="drawer-footer row wrap">
              <button
                type="button"
                className="button button--secondary"
                title="Copy observed request as JSON"
                onClick={() => void copy('details')}
              >
                Copy details
              </button>
              <button
                type="button"
                className="button button--secondary"
                onClick={() => void copy('id')}
              >
                Copy ID
              </button>
              <ActionButton>Replay request</ActionButton>
            </footer>
          </>
        ) : (
          <div className="empty-state stack">
            {error ? (
              <>
                <h2>Request unavailable</h2>
                <p role="alert">{error}</p>
                <button type="button" className="button" onClick={refresh}>
                  Retry
                </button>
              </>
            ) : (
              <p role="status">Loading request…</p>
            )}
            <Link to="/requests">Back to Requests</Link>
          </div>
        )}
      </aside>
    </div>
  )
}
