import { useEffect, useRef, useState } from 'react'
import { SegmentedControl } from '../components/Controls'
import VideoJobCard from '../components/VideoJobCard'
import {
  isPendingVideo,
  loadVideos,
  refreshVideo,
  submitVideo,
  stopVideo,
  uploadVideoConditioning,
  type VideoJob,
  type VideoGenerationRequest,
} from '../api/video'

/**
 * Manage generation settings and server-reported queue state.
 * Load history and poll only pending jobs; cleanup aborts reads/submissions and
 * clears polling timers. Provider errors never create local placeholder jobs.
 */
export default function VideoPage() {
  const [model, setModel] = useState<NonNullable<VideoGenerationRequest['model']>>('wan22-animate-14b')
  const [animationMode, setAnimationMode] = useState<NonNullable<VideoGenerationRequest['animation_mode']>>('animate')
  const [image, setImage] = useState<File | null>(null)
  const [video, setVideo] = useState<File | null>(null)
  const animate = model === 'wan22-animate-14b'
  const [prompt, setPrompt] = useState('')
  const [negative, setNegative] = useState('')
  const [duration, setDuration] = useState('8')
  const [fps, setFps] = useState('30')
  const [resolution, setResolution] =
    useState<NonNullable<VideoGenerationRequest['resolution']>>('720p')
  const [aspect, setAspect] = useState<NonNullable<VideoGenerationRequest['aspect']>>('16:9')
  const [jobs, setJobs] = useState<VideoJob[]>([])
  const [loading, setLoading] = useState(true)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [refresh, setRefresh] = useState(0)
  const active = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    loadVideos(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setJobs(data)
      })
      .catch((failure) => {
        if (!controller.signal.aborted)
          setError(
            failure instanceof Error
              ? failure.message
              : 'Could not load videos.',
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
    [],
  )

  useEffect(() => {
    const waiting = jobs.filter(isPendingVideo)
    if (!waiting.length || loading) return
    const controller = new AbortController()
    const timer = setTimeout(() => {
      Promise.all(waiting.map((job) => refreshVideo(job.id, controller.signal)))
        .then((updated) => {
          if (!controller.signal.aborted)
            setJobs((previous) =>
              previous.map(
                (job) => updated.find((next) => next.id === job.id) ?? job,
              ),
            )
        })
        .catch((failure) => {
          if (!controller.signal.aborted)
            setError(
              failure instanceof Error
                ? failure.message
                : 'Could not refresh video jobs.',
            )
        })
    }, 3000)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [jobs, loading, refresh])

  /**
   * Submit one settings snapshot and upsert the returned job by ID.
   * Only the active, non-aborted controller may update queue/error/pending state.
   */
  async function submit() {
    if (active.current) return
    const controller = new AbortController()
    active.current = controller
    setPending(true)
    setError('')
    setNotice('')
    try {
      if (animate && (!image || !video)) throw new Error('Choose an image and driving video.')
      const imageId = animate && image ? await uploadVideoConditioning(image, 'image', controller.signal) : undefined
      const videoId = animate && video ? await uploadVideoConditioning(video, 'video', controller.signal) : undefined
      const job = await submitVideo(
        {
          model,
          animation_mode: animate ? animationMode : 'animate',
          image_id: imageId,
          video_id: videoId,
          prompt,
          negative_prompt: negative,
          duration: Number(duration),
          fps: Number(fps),
          resolution,
          aspect,
        },
        controller.signal,
      )
      if (active.current === controller && !controller.signal.aborted)
        setJobs((previous) => [
          job,
          ...previous.filter((item) => item.id !== job.id),
        ])
    } catch (failure) {
      if (active.current === controller && !controller.signal.aborted)
        setError(
          failure instanceof Error
            ? failure.message
            : 'Video request failed. Please retry.',
        )
    } finally {
      if (active.current === controller) {
        active.current = null
        setPending(false)
      }
    }
  }

  /**
   * Stop waiting in this browser and ignore late submission results.
   * This does not cancel any server-accepted job; advise refreshing before retrying.
   */
  function cancel() {
    active.current?.abort()
    active.current = null
    setPending(false)
    setNotice(
      'Stopped waiting in this browser. This does not cancel any job already accepted by the server; refresh the queue before retrying.',
    )
  }

  return (
    <div className="workspace">
      <form
        className="workspace-controls"
        aria-label="Video settings"
        onSubmit={(event) => {
          event.preventDefault()
          void submit()
        }}
      >
        <label className="field">
          <span className="eyebrow">Model</span>
          <select aria-label="Video model" className="input" value={model} disabled={pending} onChange={event => {
            const value = event.target.value as NonNullable<VideoGenerationRequest['model']>
            setModel(value); setFps(value === 'wan22-animate-14b' ? '30' : '24')
            setResolution('720p'); setAspect('16:9')
          }}>
            <option value="wan22-animate-14b">Wan2.2 Animate-14B</option>
            <option value="ltx-2.5-distilled">LTX-2.5 Distilled</option>
          </select>
        </label>
        {animate && <>
          <label className="field"><span className="eyebrow">Animation mode</span>
            <select aria-label="Animation mode" className="input" value={animationMode} disabled={pending} onChange={event => setAnimationMode(event.target.value as NonNullable<VideoGenerationRequest['animation_mode']>)}>
              <option value="animate">Animation</option><option value="replace">Replacement</option>
            </select>
          </label>
          <label className="field"><span className="eyebrow">Reference image</span><input className="input" type="file" accept="image/png,image/jpeg,image/webp" required disabled={pending} onChange={event => setImage(event.target.files?.[0] ?? null)} /></label>
          <label className="field"><span className="eyebrow">Driving video</span><input className="input" type="file" accept="video/mp4" required disabled={pending} onChange={event => setVideo(event.target.files?.[0] ?? null)} /></label>
        </>}
        <label className="field">
          <span className="eyebrow">Prompt</span>
          <textarea className="input" rows={6} value={prompt} maxLength={8000} required disabled={pending} onChange={event => setPrompt(event.target.value)} placeholder="Describe the shot: subject, motion, camera, lighting…" />
        </label>
        <label className="field">
          <span className="eyebrow">Negative prompt</span>
          <input className="input" value={negative} maxLength={8000} disabled={pending} onChange={event => setNegative(event.target.value)} placeholder="Optional — things to avoid" />
        </label>
        <label className="field">
          <span className="eyebrow">Duration</span>
          <div className="input-unit"><input className="input mono" type="number" min={1} max={120} step={1} required value={duration} disabled={pending} onChange={event => setDuration(event.target.value)} /><span className="muted mono">seconds</span></div>
        </label>
        <label className="field">
          <span className="eyebrow">Frame rate</span>
          <div className="input-unit"><input className="input mono" type="number" min={1} max={120} step={1} required value={fps} disabled={pending || animate} onChange={event => setFps(event.target.value)} /><span className="muted mono">fps</span></div>
        </label>
        <SegmentedControl label="Resolution" options={animate ? ['720p'] : ['480p', '720p', '1080p']} selected={resolution} onChange={value => setResolution(value as NonNullable<VideoGenerationRequest['resolution']>)} disabled={pending} />
        <SegmentedControl label="Aspect" options={animate ? ['16:9', '9:16'] : ['16:9', '9:16', '1:1']} selected={aspect} onChange={value => setAspect(value as NonNullable<VideoGenerationRequest['aspect']>)} disabled={pending} />
        <button
          className="button button--primary"
          type="submit"
          disabled={pending || loading || !prompt.trim() || (animate && (!image || !video))}
        >
          {pending ? 'Submitting…' : 'Queue video'}
        </button>
        {pending && (
          <button className="button" type="button" onClick={cancel}>
            Cancel request
          </button>
        )}
      </form>
      <div className="workspace-results stack" aria-busy={loading || pending}>
        <div className="row">
          <h2 className="eyebrow">Queue</h2>
          <span className="mono muted">{jobs.filter(isPendingVideo).length} pending · {jobs.filter(job => job.status === 'Done').length} done</span>
          <button
            className="button"
            disabled={loading || pending}
            onClick={() => {
              setLoading(true)
              setError('')
              setRefresh((value) => value + 1)
            }}
          >
            Refresh queue
          </button>
        </div>
        {error && <p role="alert" className="error-panel">{error}</p>}
        {notice && <p role="status">{notice}</p>}
        {loading ? (
          <p role="status">Loading video jobs…</p>
        ) : (
          !jobs.length && <p>No video jobs.</p>
        )}
        {jobs.map((job) => (
          <VideoJobCard key={job.id} job={job} onCancel={() => {
            void stopVideo(job.id).then(() => setRefresh(value => value + 1))
              .catch(failure => setError(failure instanceof Error ? failure.message : 'Cancellation failed.'))
          }} />
        ))}
      </div>
    </div>
  )
}
