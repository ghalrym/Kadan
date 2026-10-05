import { Fragment } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router'
import { navigation } from '../data/navigation'

function ResourceMeters() {
  return (
    <div className="resource-meters" aria-label="Sample server metrics">
      <div className="server-status">
        Sample metrics
      </div>
      {[
        { label: 'GPU 0 · VRAM', used: '18.6', total: 24, value: 18.6 },
        { label: 'GPU 1 · VRAM', used: '14.8', total: 24, value: 14.8 },
        { label: 'RAM', used: '196.0', total: 512, value: 196 },
      ].map((meter) => (
        <div className="resource-meter" key={meter.label}>
          <div className="resource-label">
            <span>{meter.label}</span>
            <span>{meter.used}</span>
            <span className="faint">/ {meter.total} GB</span>
          </div>
          <meter
            className={meter.label === 'RAM' ? 'meter meter--blue' : 'meter'}
            min={0}
            max={meter.total}
            value={meter.value}
            aria-label={meter.label}
          />
        </div>
      ))}
    </div>
  )
}

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
