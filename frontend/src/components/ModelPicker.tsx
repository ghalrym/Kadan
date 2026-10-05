import { useEffect, useRef, useState } from 'react'
import type { ModelStatus } from '../api/generated'

/** Match the native select chevron so the custom picker sits flush with other model rows. */
function Chevron() {
  return <svg className="model-picker-chevron" width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="m3 4.5 3 3 3-3" /></svg>
}

/**
 * An in-flow dropdown group with sibling choice/download controls, rather than
 * interactive children inside native options. Escape restores trigger focus;
 * arrows/Home/End move between enabled controls, Tab and outside clicks dismiss.
 */
export function ModelPicker({ models, current, selectedId, pending, downloading, choose, download, pickerId = 'LLM', label = 'Language model' }: {
  pickerId?: string; label?: string; models: ModelStatus[]; current?: ModelStatus; selectedId: string | null; pending: boolean; downloading: boolean;
  choose: (model: ModelStatus) => void; download: (model: ModelStatus) => void;
}) {
  const [open, setOpen] = useState(false)
  const wrapper = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const popup = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    popup.current?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus()
    const outside = (event: PointerEvent) => { if (!wrapper.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [open])
  function dismiss() { setOpen(false); trigger.current?.focus() }
  return <div className="model-picker" ref={wrapper} onBlur={event => {
    if (!event.relatedTarget || !event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false)
  }} onKeyDown={event => {
    if (event.key === 'Escape' && open) { event.preventDefault(); event.stopPropagation(); dismiss() }
    if (!open || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const controls = Array.from(popup.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])
    const index = controls.indexOf(document.activeElement as HTMLButtonElement)
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? controls.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + controls.length) % controls.length
    controls[next]?.focus()
  }}>
    <button type="button" id={`model-${pickerId}`} className="input model-picker-trigger" ref={trigger} aria-expanded={open} aria-controls={`${pickerId}-model-picker`} disabled={!models.length} onClick={() => setOpen(value => !value)} onKeyDown={event => {
      if (!open && ['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); setOpen(true) }
    }}>
      <span>{current ? (current.display_name ?? current.repo_id.split('/')[1]) : `Choose a ${label.toLowerCase()}`}</span><Chevron />
    </button>
    {open && <div id={`${pickerId}-model-picker`} role="group" aria-label={`${label} options`} className="model-picker-options" ref={popup}>
      {models.map(model => <div className="model-picker-option" key={model.id}>
        <button type="button" className="model-picker-choice" disabled={pending || model.status !== 'complete'} aria-pressed={model.id === selectedId} onClick={() => { choose(model); dismiss() }}>
          <span>{(model.display_name ?? model.repo_id.split('/')[1])}</span>
          <small><span className="model-picker-size">{model.estimated_bytes ? `${(model.estimated_bytes / 1e9).toFixed(1)} GB` : 'Checking size'}</span><span className={`model-picker-status model-picker-status--${model.status}`}>{model.id === selectedId ? 'Selected' : model.status.replaceAll('_', ' ')}</span></small>
        </button>
        {model.status !== 'complete' && <button type="button" className="button model-download-icon" disabled={pending || downloading} aria-label={`${model.status === 'failed' || model.status === 'cancelled' ? 'Retry download' : 'Download'} ${(model.display_name ?? model.repo_id.split('/')[1])}`} title={`Download ${(model.display_name ?? model.repo_id.split('/')[1])}`} onClick={() => { download(model); dismiss() }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5" /></svg>
        </button>}
      </div>)}
    </div>}
  </div>
}

