import { MILESTONES, formatDuration, type ImageProgressView } from '../api/imageProgress'
import { MediaPlaceholder, type Aspect } from './Media'

/**
 * Show shared worker progress and its estimated remaining time alongside
 * the browser request’s elapsed time.
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
    <section className="image-progress" aria-label="Shared worker activity">
      <span className="eyebrow">Shared worker activity</span>
      <p className="muted">This may be another request. Your images are pending until delivered.</p>
      {view.stale && <p role="status">Worker updates unavailable. Showing last known activity.</p>}
      <div className="image-progress__head">
        <strong role="status">{view.title}</strong>
        <span className="mono muted">{percent === null ? '' : `${percent}%`}</span>
      </div>
      <div
        className={`image-progress__bar${percent === null ? ' image-progress__bar--indeterminate' : ''}`}
        role="progressbar"
        aria-label="Worker job progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
      >
        <span style={percent === null ? undefined : { width: `${percent}%` }} />
      </div>
      <div className="image-progress__time mono faint">
        <span>Request elapsed: {formatDuration(elapsed)}</span>
        {remaining !== null && !view.stale && <span>Worker ~{formatDuration(remaining)} left</span>}
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
