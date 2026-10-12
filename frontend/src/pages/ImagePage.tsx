import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ModeNavigation, SegmentedControl, UploadPlaceholder } from '../components/Controls'
import { MediaPlaceholder, type Aspect } from '../components/Media'
import { ImageProgressPanel, PendingImageSet } from '../components/ImageProgress'
import { imageHistory, requestImages, type ImageOptions } from '../api/images'
import { describeImageProgress, estimateRemaining, type ImageProgressView } from '../api/imageProgress'
import { usePolling } from '../hooks/usePolling'
import type { MetricsResponse } from '../api/generated/types.gen'
import type { ImageSet } from '../api/generated/types.gen'

const ASPECTS: Record<NonNullable<ImageOptions['aspect']>, Aspect> = {
  '1:1': 'square',
  '4:3': 'landscape',
  '3:4': 'portrait',
  '16:9': 'wide',
}

type Run = { prompt: string; aspect: Aspect; count: number; started: number }
type Tracked = {
  progress: MetricsResponse['inference_progress']
  view: ImageProgressView | null
  start: { ns: number; fraction: number } | null
}

/**
 * Select generate/edit mode and remount its workspace when the mode changes.
 */
export default function ImagePage({ edit = false }: { edit?: boolean }) {
  return <ImageWorkspace key={String(edit)} edit={edit} />
}

/**
 * Own mode-specific settings, API history and a single active image request.
 * Abort history and submission work on cleanup; show provider or metadata-only
 * errors without manufacturing image previews or claiming uploads.
 */
function ImageWorkspace({ edit }: { edit: boolean }) {
  const [busy, setBusy] = useState(false)
  const { data: metrics } = usePolling<MetricsResponse>('/v1/metrics', busy ? 1000 : 3000)
  const progress = metrics?.inference_progress
  const [run, setRun] = useState<Run | null>(null)
  const [tracked, setTracked] = useState<Tracked>({ progress: undefined, view: null, start: null })
  const [now, setNow] = useState(() => Date.now())
  // Fold each new metrics snapshot into the run's progress view during render.
  if (run && tracked.progress !== progress) {
    const view = describeImageProgress(progress, run.count, tracked.view)
    setTracked({
      progress,
      view,
      start: tracked.start ?? (view.fraction !== null && view.observedNs !== null ? { ns: view.observedNs, fraction: view.fraction } : null),
    })
  }
  const view = run ? tracked.view ?? describeImageProgress(progress, run.count, null) : null
  const remaining = view?.fraction != null && view.observedNs !== null
    ? estimateRemaining(tracked.start, { ns: view.observedNs, fraction: view.fraction })
    : null
  useEffect(() => {
    if (!run) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [run])
  const [prompt, setPrompt] = useState('')
  const [aspect, setAspect] = useState<NonNullable<ImageOptions['aspect']>>('1:1')
  const [count, setCount] = useState<1 | 2 | 4>(4)
  const [seed, setSeed] = useState('')
  const source = '' // Upload transport is not implemented; keep the original disabled placeholder.
  const [strength, setStrength] = useState(65)
  const [error, setError] = useState('')
  const [historyError, setHistoryError] = useState('')
  const [history, setHistory] = useState<ImageSet[]>([])
  const [generated, setGenerated] = useState<ImageSet[]>([])
  const images = [...generated, ...history]
  const [loading, setLoading] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const active = useRef<AbortController | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    imageHistory(controller.signal)
      .then((items) => { if (!controller.signal.aborted) setHistory(items) })
      .catch((error: unknown) => {
        if (!controller.signal.aborted)
          setHistoryError(
            error instanceof Error ? error.message : 'History unavailable.',
          )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [refresh])
  useEffect(
    () => () => {
      active.current?.abort()
      active.current = null
    },
    [edit],
  )
  /**
   * Submit normalized prompt/seed settings and optional reference/strength.
   * Prevent overlapping submissions, surface cancellation/errors only while this
   * controller is current, and leave results empty when no image files exist.
   */
  async function submit(event: FormEvent) {
    event.preventDefault()
    if (active.current || edit) return
    const controller = new AbortController()
    active.current = controller
    setBusy(true)
    setError('')
    setTracked({ progress: undefined, view: null, start: null })
    setRun({ prompt: prompt.trim(), aspect: ASPECTS[aspect], count, started: Date.now() })
    setNow(Date.now())
    try {
      const image = await requestImages(
        {
          prompt: prompt.trim(),
          aspect,
          count,
          seed: seed === '' ? null : Number(seed),
        },
        controller.signal,
        edit ? { image: source.trim(), strength: strength / 100 } : undefined,
      )
      if (active.current === controller && !controller.signal.aborted)
        setGenerated(items => [image, ...items])
    } catch (error) {
      if (active.current === controller)
        setError(
          controller.signal.aborted
            ? 'Request cancelled. Cancellation stops this browser request.'
            : error instanceof Error
              ? error.message
              : 'Cannot reach the image API. Check the backend and retry.',
        )
    } finally {
      if (active.current === controller) {
        active.current = null
        setBusy(false)
        setRun(null)
      }
    }
  }
  return (
    <div className="workspace">
      <form className="workspace-controls" aria-label="Image settings" onSubmit={submit}>
        <ModeNavigation label="Image mode" options={[{ label: 'Generate from text', to: '/image' }, { label: 'Edit an image', to: '/image/edit' }]} />
        {edit && <div className="field"><span className="eyebrow">Source image</span><UploadPlaceholder title="Drop an image or click to upload" caption="PNG, JPG, WEBP · up to 20 MB" /></div>}
        <label className="field"><span className="eyebrow">{edit ? 'Describe the edit' : 'Prompt'}</span>
          <textarea className="input" value={prompt} onChange={e => setPrompt(e.target.value)} rows={5} required maxLength={8000} disabled={busy} placeholder={edit ? 'e.g. Replace the background with a sunlit studio, keep the subject unchanged' : 'Describe the image you want…'} />
        </label>
        <div className="image-options">
          <SegmentedControl label="Aspect" options={['1:1', '4:3', '3:4', '16:9']} selected={aspect} onChange={value => setAspect(value as NonNullable<ImageOptions['aspect']>)} disabled={busy} />
          <SegmentedControl label="Images" options={['1', '2', '4']} selected={String(count)} onChange={value => setCount(Number(value) as 1 | 2 | 4)} disabled={busy} />
        </div>
        {edit && <label className="field"><span className="row"><span className="eyebrow">Edit strength</span><span className="push-right mono muted">{strength}%</span></span><input type="range" min={0} max={100} value={strength} onChange={e => setStrength(Number(e.target.value))} disabled={busy} /></label>}
        <label className="field"><span className="eyebrow">Seed</span><input className="input mono" type="number" min={0} max={Number.MAX_SAFE_INTEGER} step={1} value={seed} onChange={e => setSeed(e.target.value)} placeholder="Random" disabled={busy} /></label>
        <button className="button button--primary" disabled={busy || !prompt.trim() || edit}>{busy ? 'Generating…' : edit ? 'Apply edit' : `Generate ${count} image${count === 1 ? '' : 's'}`}</button>
        {run && view && <ImageProgressPanel view={view} elapsed={(now - run.started) / 1000} remaining={remaining} />}
        {busy && <button type="button" className="button" onClick={() => active.current?.abort()}>Cancel request</button>}
        {error && <p role="alert" className="error-panel">{error}</p>}
      </form>
      <div className="workspace-results image-gallery" aria-label="Image gallery" aria-busy={loading}>
        {run && view && <PendingImageSet view={view} prompt={run.prompt} aspect={run.aspect} count={run.count} />}
        {loading && <p role="status">Loading images…</p>}
        {historyError && <div role="alert" className="error-panel">{historyError}<button className="button" onClick={() => { setLoading(true); setHistoryError(''); setRefresh(value => value + 1) }}>Retry</button></div>}
        {!loading && !historyError && !run && images.length === 0 && <section className="stack" aria-label="Empty image gallery"><header className="image-set-heading"><span className="eyebrow">Images</span></header><div className="image-grid">{Array.from({ length: 4 }, (_, i) => <MediaPlaceholder key={i} aspect="square" label="No image" />)}</div></section>}
        {images.map(item => <section className="stack" key={item.id}><header className="image-set-heading"><span className="badge type-Image">{item.mode}</span><p>{item.prompt}</p><span className="mono faint">{item.meta}</span></header><div className="image-grid">{item.images_base64?.length && item.mime_type === 'image/png'
          ? item.images_base64.map((data, index) => <img key={index} className="generated-image" src={`data:image/png;base64,${data}`} alt={`${item.prompt} — image ${index + 1}`} />)
          : item.seeds.map((seed, index) => <MediaPlaceholder key={`${seed}-${index}`} aspect={item.aspect} label="Image unavailable" />)}</div></section>)}
      </div>
    </div>
  )
}
