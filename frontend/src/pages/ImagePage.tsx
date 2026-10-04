import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ModeNavigation } from '../components/Controls'
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
  const [aspect, setAspect] = useState<ImageOptions['aspect']>('1:1')
  const [count, setCount] = useState<1 | 2 | 4>(1)
  const [seed, setSeed] = useState('')
  const [source, setSource] = useState('')
  const [strength, setStrength] = useState(65)
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
    if (active.current) return
    const controller = new AbortController()
    active.current = controller
    setBusy(true)
    setError('')
    try {
      await requestImages(
        {
          prompt: prompt.trim(),
          aspect,
          count,
          seed: seed === '' ? null : Number(seed),
        },
        controller.signal,
        edit ? { image: source.trim(), strength: strength / 100 } : undefined,
      )
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
      <form
        className="workspace-controls"
        aria-label="Image settings"
        onSubmit={submit}
      >
        <ModeNavigation
          label="Image mode"
          options={[
            { label: 'Generate from text', to: '/image' },
            { label: 'Edit an image', to: '/image/edit' },
          ]}
        />
        <p role="note">
          No image provider is configured yet. These controls send requests to
          the API, which currently reports generation and editing as
          unavailable.
        </p>
        {edit && (
          <label className="field">
            <span className="eyebrow">Source image reference</span>
            <input
              className="input"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              required
              maxLength={2048}
              disabled={busy}
              placeholder="Image reference"
            />
            <span className="muted">
              Uploads and fetching source images require an image provider.
            </span>
          </label>
        )}
        <label className="field">
          <span className="eyebrow">
            {edit ? 'Describe the edit' : 'Prompt'}
          </span>
          <textarea
            className="input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={5}
            required
            maxLength={8000}
            disabled={busy}
          />
        </label>
        <label className="field">
          <span className="eyebrow">Aspect</span>
          <select
            className="input"
            value={aspect}
            onChange={(e) =>
              setAspect(e.target.value as ImageOptions['aspect'])
            }
            disabled={busy}
          >
            {['1:1', '4:3', '3:4', '16:9'].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="eyebrow">Images</span>
          <select
            className="input"
            value={count}
            onChange={(e) => setCount(Number(e.target.value) as 1 | 2 | 4)}
            disabled={busy}
          >
            {[1, 2, 4].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        {edit && (
          <label className="field">
            <span className="eyebrow">Edit strength {strength}%</span>
            <input
              type="range"
              min={0}
              max={100}
              value={strength}
              onChange={(e) => setStrength(Number(e.target.value))}
              disabled={busy}
            />
          </label>
        )}
        <label className="field">
          <span className="eyebrow">Seed</span>
          <input
            className="input"
            type="number"
            min={0}
            max={Number.MAX_SAFE_INTEGER}
            step={1}
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            placeholder="Random"
            disabled={busy}
          />
        </label>
        <button
          className="button button--primary"
          disabled={busy || !prompt.trim() || (edit && !source.trim())}
        >
          {busy
            ? 'Sending…'
            : edit
              ? 'Apply edit'
              : `Generate ${count} image${count === 1 ? '' : 's'}`}
        </button>
        {busy && (
          <button
            type="button"
            className="button"
            onClick={() => active.current?.abort()}
          >
            Cancel request
          </button>
        )}
        {error && <p role="alert">{error}</p>}
      </form>
      <section
        className="workspace-results image-gallery"
        aria-label="Image history"
        aria-busy={loading}
      >
        <h2>Image history</h2>
        {loading ? (
          <p>Loading history…</p>
        ) : historyError ? (
          <>
            <p role="alert">{historyError}</p>
            <button
              className="button"
              onClick={() => {
                setLoading(true)
                setHistoryError('')
                setRefresh((value) => value + 1)
              }}
            >
              Retry history
            </button>
          </>
        ) : history.length === 0 ? (
          <p>
            No generated images. An image provider must be implemented before
            images can be created.
          </p>
        ) : (
          history.map((item) => (
            <article key={item.id}>
              <p>{item.prompt}</p>
              <p>{item.meta}</p>
              <p>Image files are unavailable.</p>
            </article>
          ))
        )}
      </section>
    </div>
  )
}
