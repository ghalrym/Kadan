import { Fragment } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router'
import { navigation } from '../data/navigation'
import type { MetricsResponse } from '../api/generated/types.gen'
import { usePolling } from '../hooks/usePolling'

/** Poll observed host/device memory; unavailable probes remain explicit.
 * Values include other processes and do not establish model or GPU readiness.
 */
function ResourceMeters() {
  const { data, error, refresh } = usePolling<MetricsResponse>(
    '/v1/metrics',
    5000,
  )
  return (
    <div className="resource-meters" aria-label="Observed server memory">
      <div className="server-status">
        <span
          className={`status-dot ${data ? 'success' : error ? 'error' : 'muted'}`}
        />
        {data
          ? 'API online'
          : error
            ? 'Metrics unavailable'
            : 'Loading metrics…'}
      </div>
      {error && (
        <div role="alert">
          {error}{' '}
          <button type="button" onClick={refresh}>
            Retry
          </button>
        </div>
      )}
      {data?.resources.map((meter) => (
        <div className="resource-meter" key={meter.label}>
          <div className="resource-label">
            <span>{meter.label}</span>
            <span>{meter.used.toFixed(1)}</span>
            <span className="faint">/ {meter.total.toFixed(1)} GiB</span>
          </div>
          <meter
            className={
              meter.label === 'Host RAM' ? 'meter meter--blue' : 'meter'
            }
            min={0}
            max={meter.total}
            value={meter.used}
            aria-label={meter.label}
          />
        </div>
      ))}
      {data?.resource_errors.map((message) => (
        <p className="faint" key={message}>
          {message}
        </p>
      ))}
    </div>
  )
}

/** Render shared navigation and the route outlet alongside live memory samples. */
export default function AppLayout() {
  const { pathname } = useLocation()
  const page = navigation.find(
    (item) => pathname === item.path || pathname.startsWith(`${item.path}/`),
  )
  return (
    <div className="app-shell">
      <a className="skip-link" href="#page-content">
        Skip to content
      </a>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <img src="/kadan.svg" width="24" height="24" alt="" />
          </div>
          <div>
            <strong>Kadan</strong>
            <span className="mono muted">v0.0.1</span>
          </div>
        </div>
        <nav className="main-navigation" aria-label="Main navigation">
          {navigation.map((item, index) => (
            <Fragment key={item.path}>
              {item.group && <div className="nav-group">{item.group}</div>}
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `nav-link${isActive ? ' active' : ''}`
                }
              >
                <span className="nav-number">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <span>{item.label}</span>
              </NavLink>
            </Fragment>
          ))}
        </nav>
        <ResourceMeters />
      </aside>
      <main className="main-content">
        <header className="page-header">
          <h1>{page?.label ?? 'Page not found'}</h1>
        </header>
        <div className="page-content" id="page-content" tabIndex={-1}>
          <Outlet />
        </div>
      </main>
    </div>
  )
}
