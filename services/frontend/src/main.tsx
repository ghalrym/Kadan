import { lazy, StrictMode, Suspense, useLayoutEffect, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router'

/** Kadan is the home page; Tsuchi and Hikari are pages under /tsuchi and /hikari.
 * Each section loads on first visit and keeps its own layout and stylesheet.
 */
const Kadan = lazy(() => import('./kadan'))
const Tsuchi = lazy(() => import('./tsuchi'))
const Hikari = lazy(() => import('./hikari'))

type Section = 'kadan' | 'tsuchi' | 'hikari'
const titles: Record<Section, string> = { kadan: 'Kadan', tsuchi: 'Tsuchi', hikari: 'Hikari' }

function Page({ section, children }: { section: Section; children: ReactNode }) {
  useLayoutEffect(() => {
    // Section stylesheets are scoped to html[data-app="<section>"].
    const root = document.documentElement
    root.dataset.app = section
    if (section === 'tsuchi') root.dataset.theme = 'dark'
    else delete root.dataset.theme
    document.title = titles[section]
  }, [section])
  return <Suspense fallback={null}>{children}</Suspense>
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/tsuchi/*" element={<Page section="tsuchi"><Tsuchi /></Page>} />
        <Route path="/hikari/*" element={<Page section="hikari"><Hikari /></Page>} />
        <Route path="*" element={<Page section="kadan"><Kadan /></Page>} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
