import { Link, Outlet } from 'react-router'
import { ActionButton } from '../components/Controls'
import { requests, requestTypes, statusClass } from '../data/requests'

function RequestStats() {
  return (
    <div className="request-stats">
      {[
        { label: 'Requests / min', value: '14' },
        { label: 'p50 latency', value: '1.68', unit: 's' },
        { label: 'Error rate', value: '14.3', unit: '%' },
        { label: 'Queued jobs', value: '2' },
      ].map((stat) => (
        <div className="stat" key={stat.label}>
          <span className="eyebrow">{stat.label}</span>
          <span className="stat-value">
            {stat.value}
            <span>{stat.unit}</span>
          </span>
        </div>
      ))}
    </div>
  )
}

function RequestFilters() {
  return (
    <div className="request-filters">
      <div className="row wrap">
        <button type="button" className="filter-chip selected" disabled>
          <span className="type-dot muted" />
          All types <span className="mono muted">{requests.length}</span>
        </button>
        {requestTypes.map((type) => (
          <button type="button" className="filter-chip" disabled key={type}>
            <span className={`type-dot type-${type}`} />
            {type}
            <span className="mono muted">
              {requests.filter((request) => request.type === type).length}
            </span>
          </button>
        ))}
      </div>
      <div className="row request-search">
        <input
          className="input"
          aria-label="Search requests"
          readOnly
          placeholder="Search id, endpoint, client…"
          value=""
        />
        <ActionButton>
          <span className="status-dot success" />
          Live
        </ActionButton>
      </div>
    </div>
  )
}

export default function RequestsPage() {
  return (
    <div className="requests-page">
      <RequestStats />
      <RequestFilters />
      <div className="request-table-scroll">
        <table className="request-table">
          <caption className="sr-only">Sample request history</caption>
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
            {requests.map((request) => (
              <tr key={request.id}>
                <td className="mono muted">{request.time}</td>
                <td>
                  <span className={`request-type type-${request.type}`}>
                    <span className="type-dot" />
                    {request.type}
                  </span>
                </td>
                <td>
                  <Link
                    to={`/requests/${request.id}`}
                    className="request-link"
                    aria-label={`View ${request.type} request ${request.id}`}
                  >
                    {request.model}
                  </Link>
                </td>
                <td className={`mono ${request.ttft ? '' : 'faint'}`}>
                  {request.ttft ?? '—'}
                </td>
                <td
                  className={`mono ${request.tokensPerSecond ? '' : 'faint'}`}
                >
                  {request.tokensPerSecond ?? '—'}
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
      <Outlet />
    </div>
  )
}
