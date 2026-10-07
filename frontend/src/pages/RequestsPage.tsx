import { useState } from 'react'
import { Link, Outlet } from 'react-router'
import type {
  MetricsResponse,
  RequestsResponse,
  RequestRecord,
} from '../api/generated/types.gen'
import { usePolling } from '../hooks/usePolling'

const types: RequestRecord['type'][] = [
  'LLM',
  'Image',
  'Video',
  'TTS',
  'STT',
  'Decision',
]
/** Choose display severity from the observed HTTP outcome, not model quality. */
const statusClass = (status: number) =>
  status >= 500 ? 'error' : status >= 400 ? 'warning' : 'success'

/** Show rolling retained-sample metrics and explicitly flag truncated windows. */
function RequestStats() {
  const { data, error } = usePolling<MetricsResponse>('/v1/metrics')
  const stats = [
    {
      label: 'Requests / min',
      value: data
        ? `${data.window_truncated ? '≥' : ''}${data.requests_per_minute}`
        : '—',
    },
    {
      label: 'p50 latency',
      value:
        data?.p50_latency_seconds == null
          ? '—'
          : data.p50_latency_seconds.toFixed(3),
      unit: 's',
    },
    {
      label: 'Error rate',
      value:
        data?.error_rate_percent == null
          ? '—'
          : data.error_rate_percent.toFixed(1),
      unit: '%',
    },
    { label: 'Active requests', value: data?.active_requests ?? '—' },
  ]
  return (
    <>
      <div className="request-stats">
        {stats.map((stat) => (
          <div className="stat" key={stat.label}>
            <span className="eyebrow">{stat.label}</span>
            <span className="stat-value">
              {stat.value}
              <span>{stat.unit}</span>
            </span>
          </div>
        ))}
      </div>
      {error && <p role="alert">Metrics unavailable: {error}</p>}
      {data?.window_truncated && (
        <p role="status">
          Traffic exceeded the history limit. Latency and error rate summarize
          only retained requests.
        </p>
      )}
    </>
  )
}

/** Browse process-local observations with server-side filters and pagination.
 * Filter changes reset the page; polling may shift rows as requests complete or
 * expire. Empty, loading and failed reads stay distinct from successful results.
 */
export default function RequestsPage() {
  const [type, setType] = useState('')
  const [search, setSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const query = new URLSearchParams({
    offset: String(offset),
    limit: '25',
    search,
  })
  if (type) query.set('type', type)
  const { data, error, refresh } = usePolling<RequestsResponse>(
    `/v1/requests?${query}`,
  )
  return (
    <div className="requests-page">
      <RequestStats />
      <div className="request-filters">
        <div className="row wrap">
          {['', ...types].map((value) => (
            <button
              type="button"
              key={value}
              aria-pressed={type === value}
              className={`filter-chip ${type === value ? 'selected' : ''}`}
              onClick={() => {
                setType(value)
                setOffset(0)
              }}
            >
              <span
                className={`type-dot ${value ? `type-${value}` : 'muted'}`}
              />
              {value || 'All types'}
              <span className="mono muted">
                {value === type && data ? data.total : '—'}
              </span>
            </button>
          ))}
        </div>
        <div className="row request-search">
          <input
            className="input"
            aria-label="Search requests"
            placeholder="Search ID, endpoint or summary…"
            maxLength={128}
            value={search}
            onChange={(event) => {
              setSearch(event.target.value)
              setOffset(0)
            }}
          />
          <button
            type="button"
            className="button"
            onClick={refresh}
            aria-label="Refresh requests"
          >
            <span
              className={`status-dot ${error ? 'error' : data ? 'success' : 'muted'}`}
            />
            {error ? 'Retry' : 'Live'}
          </button>
        </div>
      </div>
      <div className="request-table-scroll">
        <table className="request-table">
          <caption className="sr-only">
            Observed generation HTTP requests, newest first
          </caption>
          <thead>
            <tr>
              <th scope="col">Time</th>
              <th scope="col">Type</th>
              <th scope="col">Model</th>
              <th scope="col">TTFT</th>
              <th scope="col">Tok/s</th>
              <th scope="col">Status</th>
              <th scope="col">Latency</th>
            </tr>
          </thead>
          <tbody>
            {(error || !data || data.requests.length === 0) && (
              <tr>
                <td colSpan={7}>
                  {error ? (
                    <p role="alert">{error}</p>
                  ) : (
                    <p role="status">
                      {data
                        ? 'No requests on this page.'
                        : 'Loading request history…'}
                    </p>
                  )}
                </td>
              </tr>
            )}
            {data?.requests.map((request) => (
              <tr key={request.id}>
                <td className="mono muted">
                  <time dateTime={request.time}>
                    {new Date(request.time).toLocaleTimeString()}
                  </time>
                </td>
                <td>
                  <span className={`request-type type-${request.type}`}>
                    <span className="type-dot" />
                    {request.type}
                  </span>
                </td>
                <td>
                  <Link
                    className="request-link"
                    to={`/requests/${request.id}`}
                    aria-label={`View request ${request.id}`}
                  >
                    {request.model ?? '—'}
                  </Link>
                </td>
                <td className="mono faint" title={request.stream_ttft_ms != null ? 'HTTP arrival to first streamed content (includes queue and text buffering)' : 'Native generation start to first generated token (excludes HTTP queue)'}>
                  {(request.stream_ttft_ms ?? request.generation_ttft_ms) != null
                    ? `${((request.stream_ttft_ms ?? request.generation_ttft_ms)! / 1000).toFixed(3)} s`
                    : '—'}
                </td>
                <td className="mono faint" title="Decode tokens per second after the first generated token; excludes prefill and cleanup">
                  {request.decode_tokens_per_second != null ? request.decode_tokens_per_second.toFixed(2) : '—'}
                </td>
                <td className={`mono ${statusClass(request.status)}`}>
                  {request.status}
                </td>
                <td className="mono">{request.latency}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(offset > 0 || (data?.total ?? 0) > 25) && (
        <div className="row wrap request-filters">
          <button
            type="button"
            className="button"
            disabled={offset === 0}
            onClick={() => setOffset((value) => Math.max(0, value - 25))}
          >
            Previous
          </button>
          <span>Page {Math.floor(offset / 25) + 1}</span>
          <button
            type="button"
            className="button"
            disabled={!data || offset + 25 >= data.total}
            onClick={() => setOffset((value) => value + 25)}
          >
            Next
          </button>
        </div>
      )}
      <Outlet />
    </div>
  )
}
