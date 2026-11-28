import { lazy, StrictMode, Suspense, useLayoutEffect, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router'

/** Kadan is the home page; Kadan Server's management pages live under /server.
 * Each section loads on first visit. Kadan Server reuses Kadan's design system.
 */
const Kadan = lazy(() => import('./kadan'))
const Server = lazy(() => import('./server'))

type Section = 'kadan' | 'server'
const titles: Record<Section, string> = { kadan: 'Kadan', server: 'Kadan Server' }

function Page({ section, children }: { section: Section; children: ReactNode }) {
  useLayoutEffect(() => {
    // Kadan's stylesheet is scoped to html[data-app="kadan"]; Kadan Server shares it.
    const root = document.documentElement
    root.dataset.app = 'kadan'
    delete root.dataset.theme
    document.title = titles[section]
  }, [section])
  return <Suspense fallback={null}>{children}</Suspense>
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/server/*" element={<Page section="server"><Server /></Page>} />
        {/* The old Hikari and Tsuchi pages became Kadan Server's AI models and agent sections. */}
        <Route path="/hikari/*" element={<Navigate to="/server/models" replace />} />
        <Route path="/tsuchi/*" element={<Navigate to="/server/agent" replace />} />
        <Route path="*" element={<Page section="kadan"><Kadan /></Page>} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
