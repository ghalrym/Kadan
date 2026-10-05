import { useState } from 'react'
import { Link, useParams } from 'react-router'
import { ActionButton } from './Controls'
import type { RequestResponse } from '../api/generated/types.gen'
import { usePolling } from '../hooks/usePolling'

const statusClass = (status: number) =>
  status >= 500 ? 'error' : status >= 400 ? 'warning' : 'success'

/** Retain the original drawer structure around actual observed summaries. */
export default function RequestDetails() {
  const { requestId } = useParams()
  const { data, error, refresh } = usePolling<RequestResponse>(
    `/v1/requests/${encodeURIComponent(requestId ?? '')}`,
  )
  const [copyError, setCopyError] = useState('')
  const request = data?.request
  async function copyId() {
    if (!request) return
    try {
      await navigator.clipboard.writeText(request.id)
      setCopyError('')
    } catch {
      setCopyError('Could not copy request ID.')
    }
  }
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
                  className={`badge status-badge ${statusClass(request.status)}`}
                >
                  {request.status}
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
                  Latency <strong>{request.latency}</strong>
                </span>
              </div>
              <section className="stack compact">
                <h3 className="eyebrow">Request summary</h3>
                <div className="detail-text">{request.prompt}</div>
              </section>
              {request.status < 400 ? (
                <section className="stack compact">
                  <h3 className="eyebrow">Response summary</h3>
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
              {copyError && <p role="alert">{copyError}</p>}
            </div>
            <footer className="drawer-footer row wrap">
              <ActionButton>Copy as cURL</ActionButton>
              <button
                type="button"
                className="button button--secondary"
                onClick={() => void copyId()}
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
