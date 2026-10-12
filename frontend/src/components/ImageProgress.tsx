import { MILESTONES, formatDuration, type ImageProgressView } from '../api/imageProgress'
import type { Aspect } from './Media'

/**
 * Summarize a running image request: an overall bar with elapsed/remaining
 * time, then the four user-facing milestones with the active one expanded.
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
    <section className="image-progress" aria-label="Generation progress">
      <div className="image-progress__head">
        <strong role="status">{view.title}</strong>
        <span className="mono muted">{percent === null ? '' : `${percent}%`}</span>
      </div>
      <div
        className={`image-progress__bar${percent === null ? ' image-progress__bar--indeterminate' : ''}`}
        role="progressbar"
        aria-label="Overall progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
      >
        <span style={percent === null ? undefined : { width: `${percent}%` }} />
      </div>
      <div className="image-progress__time mono faint">
        <span>{formatDuration(elapsed)} elapsed</span>
        {remaining !== null && view.phase !== 'finishing' && <span>~{formatDuration(remaining)} left</span>}
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

/**
 * Reserve gallery tiles for the batch in flight so finished, current and
 * waiting images are visible at a glance until the PNGs are delivered.
 */
export function PendingImageSet({
  view,
  prompt,
  aspect,
  count,
}: {
  view: ImageProgressView
  prompt: string
  aspect: Aspect
  count: number
}) {
  const generating = view.milestone === 2
  return (
    <section className="stack" aria-label="Images in progress">
      <header className="image-set-heading">
        <span className="badge type-Image">Generating</span>
        <p>{prompt}</p>
      </header>
      <div className="image-grid">
        {Array.from({ length: count }, (_, i) => {
          const index = i + 1
          const state =
            view.milestone === 3 || (generating && index < view.imageIndex)
              ? 'done'
              : generating && index === view.imageIndex
                ? 'active'
                : 'waiting'
          return (
            <div key={index} className={`media-placeholder pending-image is-${state} aspect-${aspect}`}>
              <span>
                {state === 'done'
                  ? '✓ Done'
                  : state === 'active'
                    ? view.phase === 'denoise'
                      ? `Step ${view.step} / ${view.steps}`
                      : view.detail
                    : view.milestone < 2 && index === 1
                      ? view.title
                      : 'Waiting'}
              </span>
              {state === 'active' && view.imageFraction !== null && (
                <span className="pending-image__bar" aria-hidden="true">
                  <span style={{ width: `${view.imageFraction * 100}%` }} />
                </span>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}
