import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  downloadPreview,
  modelCategories,
  tiers,
  type Tier,
  type TierId,
} from '../data/tiers'
import './SettingsPage.css'

/** Open a native modal on mount; Escape runs `close` instead of dismissing directly. */
function Modal({ title, close, children }: { title: string; close: () => void; children: ReactNode }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    dialog.current?.showModal()
  }, [])
  return (
    <dialog
      className="tier-dialog"
      ref={dialog}
      aria-labelledby="tier-dialog-title"
      onCancel={(event) => {
        event.preventDefault()
        close()
      }}
    >
      <h2 id="tier-dialog-title">{title}</h2>
      {children}
    </dialog>
  )
}

function CheckIcon() {
  return (
    <svg className="tier-card-check" width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

/** A pricing-style tile; the current tier, else the recommended one, is highlighted. */
function TierCard({ tier, current, highlighted, choose, showProgress }: {
  tier: Tier
  current: boolean
  highlighted: boolean
  choose: () => void
  showProgress: () => void
}) {
  const badge = current ? 'Current tier' : tier.recommended ? 'Recommended' : null
  return (
    <article
      className={`tier-card${highlighted ? ' tier-card--highlighted' : ''}${current ? ' tier-card--current' : ''}`}
      aria-current={current || undefined}
      aria-labelledby={`${tier.id}-name`}
    >
      {badge && <span className="tier-card-badge mono">{badge}</span>}
      <header className="tier-card-header">
        <h3 id={`${tier.id}-name`}>{tier.name}</h3>
        <p className="muted">{tier.tagline}</p>
      </header>
      <div className="tier-card-price">
        <p>
          <span className="tier-card-amount">{tier.vramGb} GB</span>
          <span className="tier-card-unit">VRAM</span>
        </p>
        <p className="muted">with {tier.ramGb} GB system RAM</p>
      </div>
      {current ? (
        <button type="button" className="button tier-card-button" onClick={showProgress}>
          View download preview
        </button>
      ) : (
        <button
          type="button"
          className={`button tier-card-button${highlighted ? ' button--primary' : ''}`}
          onClick={choose}
        >
          Choose {tier.name}
        </button>
      )}
      <div className="tier-card-includes">
        <p className="tier-card-includes-label mono">Includes</p>
        <ul>
          {modelCategories.map(({ id, shortLabel }) => (
            <li key={id}>
              <CheckIcon />
              <span>
                <span className="muted">{shortLabel}</span> {tier.models[id]}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </article>
  )
}

function DownloadProgress() {
  return (
    <ul className="tier-downloads">
      {modelCategories.map(({ id, label }) => {
        const model = downloadPreview.find((item) => item.category === id)
        if (!model) return null
        return (
          <li key={id} className={`tier-download tier-download--${model.status}`}>
            <div className="tier-download-heading">
              <span>{label}</span>
              <span className="mono tier-download-status">{model.status}</span>
            </div>
            <progress
              className="progress"
              aria-label={`${label} download progress`}
              max={100}
              value={model.percent}
            />
            <span className="mono faint">
              {model.percent}%
            </span>
          </li>
        )
      })}
    </ul>
  )
}

export default function SettingsPage() {
  const [currentTier, setCurrentTier] = useState<TierId | null>(null)
  const [confirmTier, setConfirmTier] = useState<Tier | null>(null)
  const [progressOpen, setProgressOpen] = useState(false)

  function chooseTier(tier: Tier) {
    setCurrentTier(tier.id)
    setConfirmTier(null)
    setProgressOpen(true)
  }

  const selectedTier = tiers.find((tier) => tier.id === currentTier)

  return (
    <div className="scroll-page">
      {confirmTier && (
        <Modal title={`Download ${confirmTier.name} models?`} close={() => setConfirmTier(null)}>
          <p className="muted">
            {currentTier
              ? `Switching to ${confirmTier.name} downloads the language, image, video, and speech models sized for ${confirmTier.vramGb} GB VRAM and ${confirmTier.ramGb} GB RAM.`
              : `This downloads the language, image, video, and speech models sized for ${confirmTier.vramGb} GB VRAM and ${confirmTier.ramGb} GB RAM.`}
          </p>
          <div className="tier-dialog-actions">
            <button type="button" className="button" autoFocus onClick={() => setConfirmTier(null)}>
              No
            </button>
            <button type="button" className="button button--primary" onClick={() => chooseTier(confirmTier)}>
              Yes, download
            </button>
          </div>
        </Modal>
      )}
      {progressOpen && selectedTier && (
        <Modal
          title={`${selectedTier.name} download preview`}
          close={() => setProgressOpen(false)}
        >
          <p className="muted">Example progress only. No models are being downloaded.</p>
          <DownloadProgress />
          <div className="tier-dialog-actions">
            <button type="button" className="button button--primary" onClick={() => setProgressOpen(false)}>
              Done
            </button>
          </div>
        </Modal>
      )}
      <div className="settings-layout tier-layout stack">
        <header className="tier-intro">
          <h1>Choose your tier</h1>
          <p className="muted">Pick the one that matches your GPU and RAM.</p>
        </header>
        <div className="tier-cards">
          {tiers.map((tier) => (
            <TierCard
              key={tier.id}
              tier={tier}
              current={tier.id === currentTier}
              highlighted={currentTier ? tier.id === currentTier : tier.recommended}
              choose={() => setConfirmTier(tier)}
              showProgress={() => setProgressOpen(true)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
