import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ModeNavigation, SegmentedControl } from '../components/Controls'
import { MediaPlaceholder, type Aspect } from '../components/Media'
import { ImageProgressPanel, PendingImageSet } from '../components/ImageProgress'
import { createImageRequestId, imageHistory, requestImages, type ImageOptions } from '../api/images'
import { prepareImageUpload, type ImageUpload } from '../api/imageUpload'
import { describeImageProgress, estimateRemaining, type ImageProgressView } from '../api/imageProgress'
import { usePolling } from '../hooks/usePolling'
import type { MetricsResponse } from '../api/generated/types.gen'
import type { ImageSet } from '../api/generated/types.gen'
import './ImagePage.css'

const ASPECTS: Record<NonNullable<ImageOptions['aspect']>, Aspect> = {
  '1:1': 'square',
  '4:3': 'landscape',
  '3:4': 'portrait',
  '16:9': 'wide',
}

type Run = { id: string; prompt: string; aspect: Aspect; count: number; started: number }
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
  const [run, setRun] = useState<Run | null>(null)
  const { data: metrics } = usePolling<MetricsResponse>(run ? `/v1/metrics?image_request_id=${run.id}` : '/v1/metrics', busy ? 1000 : 3000)
  const progress = run ? metrics?.inference_progress : undefined
  const [tracked, setTracked] = useState<Tracked>({ progress: undefined, view: null, start: null })
  const [now, setNow] = useState(() => Date.now())
  // Keep the last request snapshot across failed polls.
  if (run && tracked.progress !== progress) {
    const view = describeImageProgress(progress, tracked.view, run.id)
    setTracked({
      progress,
      view,
      start: (view.jobId === tracked.view?.jobId ? tracked.start : null) ?? (view.fraction !== null && view.observedNs !== null ? { ns: view.observedNs, fraction: view.fraction } : null),
    })
  }
  const view = run ? tracked.view ?? describeImageProgress(progress, null, run.id) : null
  const remaining = view && !view.stale && view.fraction != null && view.observedNs !== null
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
  const [source, setSource] = useState<ImageUpload | null>(null)
  const [mask, setMask] = useState<ImageUpload | null>(null)
  const [preparing, setPreparing] = useState<'source' | 'mask' | null>(null)
  const uploadVersion = useRef(0)
  const sourceInput = useRef<HTMLInputElement>(null)
  const maskInput = useRef<HTMLInputElement>(null)
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
      uploadVersion.current += 1
      active.current?.abort()
      active.current = null
    },
    [edit],
  )
  async function selectUpload(file: File | undefined, kind: 'source' | 'mask') {
    if (!file || active.current) return
    const version = ++uploadVersion.current
    setPreparing(kind)
    setError('')
    // Replacing a source invalidates its mask, including an in-flight mask read.
    if (kind === 'source') {
      setSource(null)
      setMask(null)
    } else {
      setMask(null)
    }
    try {
      if (kind === 'mask' && !source) throw new Error('Choose a source image first.')
      const upload = await prepareImageUpload(file, kind === 'mask' ? source! : undefined)
      if (version === uploadVersion.current) {
        if (kind === 'source') setSource(upload)
        else setMask(upload)
      }
    } catch (error) {
      if (version === uploadVersion.current)
        setError(error instanceof Error ? error.message : 'Could not prepare this image. Choose another file.')
    } finally {
      if (version === uploadVersion.current) setPreparing(null)
    }
  }
  function removeUpload(kind: 'source' | 'mask') {
    uploadVersion.current += 1
    setPreparing(null)
    setError('')
    setMask(null)
    if (kind === 'source') setSource(null)
  }
  /**
   * Submit normalized prompt/seed settings and optional reference/strength.
   * Prevent overlapping submissions, surface cancellation/errors only while this
   * controller is current, and leave results empty when no image files exist.
   */
  async function submit(event: FormEvent) {
    event.preventDefault()
    if (active.current || preparing || !prompt.trim() || (edit && !source)) return
    const requestId = createImageRequestId()
    const controller = new AbortController()
    active.current = controller
    setBusy(true)
    setError('')
    setTracked({ progress: undefined, view: null, start: null })
    const previewAspect: Aspect = edit && source
      ? source.width === source.height ? 'square'
        : source.width < source.height ? 'portrait'
          : source.width / source.height > 1.5 ? 'wide' : 'landscape'
      : ASPECTS[aspect]
    setRun({ id: requestId, prompt: prompt.trim(), aspect: previewAspect, count, started: Date.now() })
    setNow(Date.now())
    try {
      const image = await requestImages(
        {
          prompt: prompt.trim(),
          ...(edit ? {} : { aspect }),
          count,
          seed: seed === '' ? null : Number(seed),
        },
        controller.signal,
        edit && source ? { image: source.dataUrl, mask: mask?.dataUrl ?? null, strength: strength / 100 } : undefined,
        requestId,
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
    <div className="workspace image-workspace">
      <form className="workspace-controls" aria-label="Image settings" onSubmit={submit}>
        <ModeNavigation label="Image mode" options={[{ label: 'Generate from text', to: '/image' }, { label: 'Edit an image', to: '/image/edit' }]} />
        {edit && <>
          <div className="field">
            <span className="eyebrow">Source image</span>
            <input ref={sourceInput} className="sr-only" type="file" accept="image/png,image/jpeg,image/webp" aria-label="Choose source image" disabled={busy || preparing !== null} onChange={event => { void selectUpload(event.target.files?.[0], 'source'); event.target.value = '' }} />
            {source && <>
              <img className="image-upload-preview" src={source.dataUrl} alt="Source image" />
              <span className="image-upload-name muted">{source.name}</span>
              <span className="mono muted">Output: {source.width} × {source.height} px</span>
            </>}
            <div className="row wrap">
              <button type="button" className="button" disabled={busy || preparing !== null} onClick={() => sourceInput.current?.click()}>{source ? 'Replace image' : 'Choose image'}</button>
              {source && <button type="button" className="button" disabled={busy} onClick={() => removeUpload('source')}>Remove image</button>}
            </div>
            {!source && <span className="mono muted">PNG, JPEG, WEBP · up to 20 MiB</span>}
          </div>
          {source && <div className="field">
            <span className="eyebrow">Mask (optional)</span>
            <input ref={maskInput} className="sr-only" type="file" accept="image/png,image/jpeg,image/webp" aria-label="Choose mask" disabled={busy || preparing !== null} onChange={event => { void selectUpload(event.target.files?.[0], 'mask'); event.target.value = '' }} />
            {mask && <><img className="image-upload-preview" src={mask.dataUrl} alt="Edit mask" /><span className="image-upload-name muted">{mask.name}</span></>}
            <div className="row wrap">
              <button type="button" className="button" disabled={busy || preparing !== null} onClick={() => maskInput.current?.click()}>{mask ? 'Replace mask' : 'Choose mask'}</button>
              {mask && <button type="button" className="button" disabled={busy} onClick={() => removeUpload('mask')}>Remove mask</button>}
            </div>
            <span className="muted">Black preserves, white edits. Use an opaque {source.originalWidth} × {source.originalHeight} px mask.</span>
          </div>}
          {preparing && <p role="status">Preparing {preparing === 'source' ? 'image' : 'mask'}…</p>}
        </>}
        <label className="field"><span className="eyebrow">{edit ? 'Describe the result' : 'Prompt'}</span>
          <textarea className="input" value={prompt} onChange={e => setPrompt(e.target.value)} rows={5} required maxLength={8000} disabled={busy} placeholder={edit ? 'e.g. A ceramic mug on a table in a sunlit studio' : 'Describe the image you want…'} />
        </label>
        <div className={edit ? 'image-options image-options--edit' : 'image-options'}>
          {!edit && <SegmentedControl label="Aspect" options={['1:1', '4:3', '3:4', '16:9']} selected={aspect} onChange={value => setAspect(value as NonNullable<ImageOptions['aspect']>)} disabled={busy} />}
          <SegmentedControl label="Images" options={['1', '2', '4']} selected={String(count)} onChange={value => setCount(Number(value) as 1 | 2 | 4)} disabled={busy} />
        </div>
        {edit && <label className="field"><span className="row"><span className="eyebrow">Edit strength</span><span className="push-right mono muted">{strength}%</span></span><input type="range" min={0} max={100} value={strength} onChange={e => setStrength(Number(e.target.value))} disabled={busy} /><span className="row image-strength-scale muted"><span>0% preserves image</span><span>100% regenerates</span></span></label>}
        <label className="field"><span className="eyebrow">Seed</span><input className="input mono" type="number" min={0} max={Number.MAX_SAFE_INTEGER} step={1} value={seed} onChange={e => setSeed(e.target.value)} placeholder="Random" disabled={busy} /></label>
        <button className="button button--primary" disabled={busy || preparing !== null || !prompt.trim() || (edit && !source)}>{busy ? 'Waiting for images…' : edit ? 'Apply edit' : `Generate ${count} image${count === 1 ? '' : 's'}`}</button>
        {run && view && <ImageProgressPanel view={view} elapsed={(now - run.started) / 1000} remaining={remaining} />}
        {busy && <button type="button" className="button" onClick={() => active.current?.abort()}>Cancel request</button>}
        {error && <p role="alert" className="error-panel">{error}</p>}
      </form>
      <div className="workspace-results image-gallery" aria-label="Image gallery" aria-busy={loading}>
        {run && <PendingImageSet prompt={run.prompt} aspect={run.aspect} count={run.count} />}
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
