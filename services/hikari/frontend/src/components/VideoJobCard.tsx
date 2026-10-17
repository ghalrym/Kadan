import { MediaPlaceholder } from './Media'
import { isPendingVideo, type VideoJob } from '../api/video'

/**
 * Display actual job state and completed native media.
 */
export default function VideoJobCard({ job, onCancel }: { job: VideoJob; onCancel?: () => void }) {
  return (
    <article className="panel video-job">
      <div className="video-thumbnail">
        {job.status === 'Done' && job.output_url
          ? <video controls preload="metadata" src={job.output_url} aria-label="Generated video" style={{ width: '100%' }} />
          : <MediaPlaceholder aspect={job.aspect} label={job.thumbnail} />}
      </div>
      <div className="stack compact">
        <div className="row wrap">
          <span className={`badge job-${job.status.toLowerCase()}`}>
            {job.status}
          </span>
          <span className="mono faint">{job.id}</span>
          <span className="push-right mono faint">{job.time}</span>
        </div>
        <p className="job-prompt">{job.prompt}</p>
        <p className="mono muted">
          {job.duration} · {job.resolution} ·{' '}
          {job.aspect === 'wide'
            ? '16:9'
            : job.aspect === 'square'
              ? '1:1'
              : '9:16'}{' '}
          · {job.fps} fps
        </p>
        {job.error && <p role="alert">{job.error}</p>}
        {isPendingVideo(job) && onCancel && <button type="button" className="ghost" onClick={onCancel}>Cancel</button>}
        <div className="job-progress row">
          <progress
            className="progress progress--video"
            max={100}
            value={job.progress}
            aria-label={`${job.id} render progress`}
          />
          <span className="mono">{job.progressText}</span>
        </div>
      </div>
    </article>
  )
}
