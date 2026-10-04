import { useEffect, useRef, useState } from 'react'
import VideoJobCard from '../components/VideoJobCard'
import {
  isPendingVideo,
  loadVideos,
  refreshVideo,
  submitVideo,
  type VideoJob,
  type VideoGenerationRequest,
} from '../api/video'

/**
 * Manage generation settings and server-reported queue state.
 * Load history and poll only pending jobs; cleanup aborts reads/submissions and
 * clears polling timers. Provider errors never create local placeholder jobs.
 */
export default function VideoPage() {
  const [prompt, setPrompt] = useState('')
  const [negative, setNegative] = useState('')
  const [duration, setDuration] = useState('8')
  const [fps, setFps] = useState('24')
  const [resolution, setResolution] =
    useState<VideoGenerationRequest['resolution']>('720p')
  const [aspect, setAspect] = useState<VideoGenerationRequest['aspect']>('16:9')
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
      const job = await submitVideo(
        {
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
          Prompt
          <textarea
            className="input"
            rows={6}
            value={prompt}
            maxLength={8000}
            required
            onChange={(event) => setPrompt(event.target.value)}
          />
        </label>
        <label className="field">
          Negative prompt
          <input
            className="input"
            value={negative}
            maxLength={8000}
            onChange={(event) => setNegative(event.target.value)}
          />
        </label>
        <label className="field">
          Duration (seconds)
          <input
            className="input"
            type="number"
            min={1}
            max={120}
            step={1}
            required
            value={duration}
            onChange={(event) => setDuration(event.target.value)}
          />
        </label>
        <label className="field">
          Frame rate (fps)
          <input
            className="input"
            type="number"
            min={1}
            max={120}
            step={1}
            required
            value={fps}
            onChange={(event) => setFps(event.target.value)}
          />
        </label>
        <label className="field">
          Resolution
          <select
            className="input"
            value={resolution}
            onChange={(event) =>
              setResolution(
                event.target.value as VideoGenerationRequest['resolution'],
              )
            }
          >
            {['480p', '720p', '1080p'].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label className="field">
          Aspect
          <select
            className="input"
            value={aspect}
            onChange={(event) =>
              setAspect(event.target.value as VideoGenerationRequest['aspect'])
            }
          >
            {['16:9', '9:16', '1:1'].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <p className="muted">
          Video generation is not available yet. No video provider is
          configured.
        </p>
        <button
          className="button button--primary"
          type="submit"
          disabled={pending || loading || !prompt.trim()}
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
        {error && <p role="alert">{error}</p>}
        {notice && <p role="status">{notice}</p>}
        {loading ? (
          <p role="status">Loading video jobs…</p>
        ) : (
          !jobs.length && <p>No video jobs.</p>
        )}
        {jobs.map((job) => (
          <VideoJobCard key={job.id} job={job} />
        ))}
      </div>
    </div>
  )
}
