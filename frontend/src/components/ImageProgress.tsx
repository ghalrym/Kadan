import { MILESTONES, formatDuration, type ImageProgressView } from '../api/imageProgress'
import { MediaPlaceholder, type Aspect } from './Media'

/**
 * Show this request's latest progress and estimated remaining time.
 */
export function ImageProgressPanel({
  view,
  elapsed,
  remaining,
}: {
  view: ImageProgressView
  elapsed: number
  remaining: number | null
}) {
  const percent = view.fraction === null ? null : Math.round(view.fraction * 100)
  return (
    <section className="image-progress" aria-label="Image generation progress">
      <span className="eyebrow">Generation progress</span>
      <p className="muted">Your images are pending until delivered.</p>
      {view.stale && <p role="status">Waiting for updates…</p>}
      <div className="image-progress__head">
        <strong role="status">{view.title}</strong>
        <span className="mono muted">{percent === null ? '' : `${percent}%`}</span>
      </div>
      <div
        className={`image-progress__bar${percent === null ? ' image-progress__bar--indeterminate' : ''}`}
        role="progressbar"
        aria-label="Image request progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
      >
        <span style={percent === null ? undefined : { width: `${percent}%` }} />
      </div>
      <div className="image-progress__time mono faint">
        <span>Request elapsed: {formatDuration(elapsed)}</span>
        {remaining !== null && !view.stale && <span>~{formatDuration(remaining)} left</span>}
      </div>
      <ol className="image-progress__steps">
        {MILESTONES.map((label, index) => {
          const state = index < view.milestone ? 'done' : index === view.milestone ? 'active' : 'pending'
          return (
            <li key={label} className={`image-progress__step is-${state}`} aria-current={state === 'active' ? 'step' : undefined}>
              <span className="image-progress__marker" aria-hidden="true">{state === 'done' ? '✓' : ''}</span>
              <span className="image-progress__label">
                {index === 2 && view.imageCount > 1 ? `Generate ${view.imageCount} images` : label}
                {state === 'active' && <span className="image-progress__detail">{view.detail}</span>}
                {state === 'active' && view.phase === 'denoise' && view.imageFraction !== null && (
                  <span className="image-progress__substeps" aria-hidden="true">
                    <span style={{ width: `${(view.step / Math.max(view.steps, 1)) * 100}%` }} />
                  </span>
                )}
              </span>
            </li>
          )
        })}
      </ol>
    </section>
  )
}

/** Keep request tiles pending until its response delivers actual images. */
export function PendingImageSet({
  prompt,
  aspect,
  count,
}: {
  prompt: string
  aspect: Aspect
  count: number
}) {
  return (
    <section className="stack" aria-label="Images pending delivery">
      <header className="image-set-heading">
        <span className="badge type-Image">Pending</span>
        <p>{prompt}</p>
      </header>
      <div className="image-grid">
        {Array.from({ length: count }, (_, index) => (
          <MediaPlaceholder key={index} aspect={aspect} label="Waiting for images" />
        ))}
      </div>
    </section>
  )
}
