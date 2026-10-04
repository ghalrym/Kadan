import { ActionButton } from './Controls'
import { MediaPlaceholder } from './Media'
import type { VideoJob } from '../data/playground'

export default function VideoJobCard({ job }: { job: VideoJob }) {
  return (
    <article className="panel video-job">
      <div className="video-thumbnail">
        <MediaPlaceholder aspect={job.aspect} label={job.thumbnail} />
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
          {job.aspect === 'wide' ? '16:9' : '9:16'} · {job.fps} fps · Wan 2.2
          T2V 14B
        </p>
        <div className="job-progress row">
          <progress
            className="progress progress--video"
            max={100}
            value={job.progress}
            aria-label={`${job.id} render progress`}
          />
          <span className="mono">{job.progressText}</span>
        </div>
        {job.status === 'Done' && (
          <div className="row">
            <ActionButton>▶ Play</ActionButton>
            <ActionButton>Download MP4</ActionButton>
          </div>
        )}
      </div>
    </article>
  )
}
