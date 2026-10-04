import { ActionButton } from './Controls'

export type Aspect = 'square' | 'landscape' | 'wide' | 'portrait'

export function MediaPlaceholder({
  label,
  aspect = 'wide',
}: {
  label: string
  aspect?: Aspect
}) {
  return (
    <div className={`media-placeholder aspect-${aspect}`}>
      <span>{label}</span>
    </div>
  )
}

export function Waveform() {
  return (
    <div className="waveform" aria-hidden="true">
      <svg viewBox="0 0 720 48" preserveAspectRatio="none">
        {Array.from({ length: 72 }, (_, index) => {
          const height = Math.max(
            4,
            Math.round(
              Math.abs(
                Math.sin(index * 0.37 + 2.1) * 0.6 +
                  Math.sin(index * 1.3 + 4.2) * 0.3 +
                  Math.sin(index * 0.07) * 0.2,
              ) * 44,
            ),
          )
          return (
            <rect
              key={index}
              x={index * 10}
              y={(48 - height) / 2}
              width="6"
              height={height}
              rx="2"
            />
          )
        })}
      </svg>
    </div>
  )
}

export function AudioCard({
  voice,
  meta,
  script,
  time,
}: {
  voice: string
  meta: string
  script: string
  time: string
}) {
  return (
    <article className="panel audio-card">
      <div className="row">
        <ActionButton className="play-button" label="Play audio">
          ▶
        </ActionButton>
        <div className="stack compact grow">
          <strong>{voice}</strong>
          <span className="mono muted">{meta}</span>
        </div>
        <ActionButton>Download</ActionButton>
      </div>
      <Waveform />
      <div className="audio-caption">
        <span className="mono muted">0:00 / {time}</span>
        <span className="muted">“{script}”</span>
      </div>
    </article>
  )
}
