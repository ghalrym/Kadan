import { Link, useParams } from 'react-router'
import type { RequestResponse } from '../api/generated/types.gen'
import { usePolling } from '../hooks/usePolling'

/** Display a polled retained observation, including expiry/error states.
 * Summary fields contain counts and HTTP metadata, never original payload text.
 */
export default function RequestDetails() {
  const { requestId } = useParams()
  const { data, error, refresh } = usePolling<RequestResponse>(
    `/v1/requests/${encodeURIComponent(requestId ?? '')}`,
  )
  const request = data?.request
  return (
    <div className="detail-overlay">
      <Link
        to="/requests"
        className="detail-backdrop"
        aria-label="Close request details"
        tabIndex={-1}
      />
      <aside className="request-drawer" aria-label="Request details">
        <header className="drawer-header row wrap">
          <h2>Request details</h2>
          <Link to="/requests" className="button push-right">
            Close
          </Link>
        </header>
        <div className="drawer-content stack">
          {error && (
            <div role="alert">
              {error}{' '}
              <button type="button" onClick={refresh}>
                Retry
              </button>
            </div>
          )}
          {!request && !error && <p role="status">Loading request…</p>}
          {request && (
            <>
              <p className="mono">{request.id}</p>
              <p>
                <time dateTime={request.time}>
                  {new Date(request.time).toLocaleString()}
                </time>
              </p>
              <p className="endpoint">POST {request.endpoint}</p>
              <div className="detail-metrics mono">
                <span>
                  HTTP <strong>{request.status}</strong>
                </span>
                <span>
                  Latency <strong>{request.latency}</strong>
                </span>
              </div>
              <p>
                Requested model:{' '}
                {request.model ?? 'Not recorded (default or non-catalog ID)'}
              </p>
              <section className="stack compact">
                <h3 className="eyebrow">Request summary</h3>
                <p className="detail-text">{request.prompt}</p>
              </section>
              <section className="stack compact">
                <h3 className="eyebrow">Response summary</h3>
                <p className="detail-text">{request.output}</p>
              </section>
              <details>
                <summary>Measured record</summary>
                <pre>{JSON.stringify(request, null, 2)}</pre>
              </details>
            </>
          )}
        </div>
      </aside>
    </div>
  )
}
