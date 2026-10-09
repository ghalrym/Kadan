import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ModeNavigation, SegmentedControl, UploadPlaceholder } from '../components/Controls'
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
  const [count, setCount] = useState<1 | 2 | 4>(1)
  const [seed, setSeed] = useState('')
  const source = '' // Upload transport is not implemented; keep the original disabled placeholder.
  const [strength, setStrength] = useState(65)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [historyError, setHistoryError] = useState('')
  const [history, setHistory] = useState<ImageSet[]>([])
  const [loading, setLoading] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const active = useRef<AbortController | null>(null)
  const completed = useRef<ImageSet[]>([])
  useEffect(() => {
    const controller = new AbortController()
    imageHistory(controller.signal)
      .then((items) => {
        if (controller.signal.aborted) return
        // History may have been read before a submission finished. Keep this
        // workspace's completed results authoritative over that older snapshot.
        const ids = new Set(completed.current.map(item => item.id))
        setHistory([...completed.current, ...items.filter(item => !ids.has(item.id))])
      })
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
      if (active.current === controller && !controller.signal.aborted) {
        completed.current = [image, ...completed.current.filter(item => item.id !== image.id)]
        setHistory(items => [image, ...items.filter(item => item.id !== image.id)])
        setHistoryError('')
      }
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
        <button className="button button--primary" disabled={busy || !prompt.trim() || edit}>{busy ? 'Sending…' : edit ? 'Apply edit' : `Generate ${count} image${count === 1 ? '' : 's'}`}</button>
        {busy && <button type="button" className="button" onClick={() => active.current?.abort()}>Cancel request</button>}
        {error && <p role="alert" className="error-panel">{error}</p>}
      </form>
      <div className="workspace-results image-gallery" aria-label="Image gallery" aria-busy={loading}>
        {loading && <p role="status">Loading images…</p>}
        {historyError && <div role="alert" className="error-panel">{historyError}<button className="button" onClick={() => { setLoading(true); setHistoryError(''); setRefresh(value => value + 1) }}>Retry</button></div>}
        {!loading && !historyError && history.length === 0 && <section className="stack" aria-label="Empty image gallery"><header className="image-set-heading"><span className="eyebrow">Images</span></header><div className="image-grid">{Array.from({ length: 4 }, (_, i) => <MediaPlaceholder key={i} aspect="square" label="No image" />)}</div></section>}
        {history.map(item => <section className="stack" key={item.id}><header className="image-set-heading"><span className="badge type-Image">{item.mode}</span><p>{item.prompt}</p><span className="mono faint">{item.meta}</span></header><div className="image-grid">{item.urls?.length ? item.urls.map((url, index) => <a key={url} href={url} target="_blank" rel="noreferrer"><img className={`generated-image aspect-${item.aspect}`} src={url} alt={`${item.prompt} · Image ${index + 1}`} loading="lazy" /></a>) : <MediaPlaceholder aspect={item.aspect} label="Image unavailable" />}</div></section>)}
      </div>
    </div>
  )
}
