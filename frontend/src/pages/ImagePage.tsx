import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ModeNavigation, SegmentedControl } from '../components/Controls'
import { MediaPlaceholder } from '../components/Media'
import { imageHistory, requestImages, type ImageOptions } from '../api/images'
import type { ImageSet } from '../api/generated/types.gen'

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
  const [prompt, setPrompt] = useState('')
  const [aspect, setAspect] = useState<NonNullable<ImageOptions['aspect']>>('1:1')
  const [count, setCount] = useState<1 | 2 | 4>(4)
  const [seed, setSeed] = useState('')
  const [source, setSource] = useState('')
  const [sourceName, setSourceName] = useState('')
  const upload = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [historyError, setHistoryError] = useState('')
  const [history, setHistory] = useState<ImageSet[]>([])
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
    if (active.current || (edit && !source)) return
    const controller = new AbortController()
    active.current = controller
    setBusy(true)
    setError('')
    try {
      const result = await requestImages(
        {
          prompt: prompt.trim(),
          aspect,
          count,
          seed: seed === '' ? null : Number(seed),
        },
        controller.signal,
        edit ? { image: source } : undefined,
      )
      if (active.current === controller && !controller.signal.aborted)
        setHistory(items => [result, ...items])
    } catch (error) {
      if (active.current === controller)
        setError(
          controller.signal.aborted
            ? 'Request cancelled. Native generation is being cancelled.'
            : error instanceof Error
              ? error.message
              : 'Cannot reach the image API. Check the backend and retry.',
        )
    } finally {
      if (active.current === controller) {
        active.current = null
        setBusy(false)
      }
    }
  }
  return (
    <div className="workspace">
      <form className="workspace-controls" aria-label="Image settings" onSubmit={submit}>
        <ModeNavigation label="Image mode" options={[{ label: 'Generate from text', to: '/image' }, { label: 'Edit an image', to: '/image/edit' }]} />
        {edit && <div className="field"><span className="eyebrow">Source image</span><input hidden ref={upload} type="file" accept="image/png,image/jpeg,image/webp" disabled={busy} onChange={event => {
          const file = event.target.files?.[0]
          setSource(''); setSourceName('')
          if (!file) return
          if (file.size > 20 * 1024 * 1024) { setError('Source image exceeds 20 MB'); return }
          const reader = new FileReader()
          reader.onload = () => { setSource(String(reader.result)); setSourceName(file.name); setError('') }
          reader.onerror = () => setError('Cannot read this image')
          reader.readAsDataURL(file)
        }} /><button type="button" className="upload-placeholder" disabled={busy} onClick={() => upload.current?.click()}><span>{sourceName || 'Click to upload an image'}</span><span className="mono faint">PNG, JPG, WEBP · up to 20 MB</span></button></div>}
        <label className="field"><span className="eyebrow">{edit ? 'Describe the edit' : 'Prompt'}</span>
          <textarea className="input" value={prompt} onChange={e => setPrompt(e.target.value)} rows={5} required maxLength={8000} disabled={busy} placeholder={edit ? 'e.g. Replace the background with a sunlit studio, keep the subject unchanged' : 'Describe the image you want…'} />
        </label>
        <div className="image-options">
          <SegmentedControl label="Aspect" options={['1:1', '4:3', '3:4', '16:9']} selected={aspect} onChange={value => setAspect(value as NonNullable<ImageOptions['aspect']>)} disabled={busy} />
          <SegmentedControl label="Images" options={['1', '2', '4']} selected={String(count)} onChange={value => setCount(Number(value) as 1 | 2 | 4)} disabled={busy} />
        </div>

        <label className="field"><span className="eyebrow">Seed</span><input className="input mono" type="number" min={0} max={Number.MAX_SAFE_INTEGER} step={1} value={seed} onChange={e => setSeed(e.target.value)} placeholder="Random" disabled={busy} /></label>
        <button className="button button--primary" disabled={busy || !prompt.trim() || (edit && !source)}>{busy ? 'Sending…' : edit ? 'Apply edit' : `Generate ${count} image${count === 1 ? '' : 's'}`}</button>
        {busy && <button type="button" className="button" onClick={() => active.current?.abort()}>Cancel request</button>}
        {error && <p role="alert" className="error-panel">{error}</p>}
      </form>
      <div className="workspace-results image-gallery" aria-label="Image gallery" aria-busy={loading}>
        {loading && <p role="status">Loading images…</p>}
        {historyError && <div role="alert" className="error-panel">{historyError}<button className="button" onClick={() => { setLoading(true); setHistoryError(''); setRefresh(value => value + 1) }}>Retry</button></div>}
        {!loading && !historyError && history.length === 0 && <section className="stack" aria-label="Empty image gallery"><header className="image-set-heading"><span className="eyebrow">Images</span></header><div className="image-grid">{Array.from({ length: 4 }, (_, i) => <MediaPlaceholder key={i} aspect="square" label="No image" />)}</div></section>}
        {history.map(item => <section className="stack" key={item.id}><header className="image-set-heading"><span className="badge type-Image">{item.mode}</span><p>{item.prompt}</p><span className="mono faint">{item.meta}</span></header><div className="image-grid">{item.seeds.map((seed, index) => item.urls?.[index] ? <a key={seed} href={item.urls[index]} target="_blank" rel="noreferrer"><img className="generated-image" src={item.urls[index]} alt={item.prompt} /></a> : <MediaPlaceholder key={seed} aspect={item.aspect} label="Image unavailable" />)}</div></section>)}
      </div>
    </div>
  )
}
