import { ActionButton, Field, SegmentedControl } from '../components/Controls'
import VideoJobCard from '../components/VideoJobCard'
import { videoJobs } from '../data/playground'

export default function VideoPage() {
  return (
    <div className="workspace">
      <aside className="workspace-controls" aria-label="Video settings">
        <Field
          label="Prompt"
          multiline
          rows={6}
          placeholder="Describe the shot: subject, motion, camera, lighting…"
        />
        <Field
          label="Negative prompt"
          placeholder="Optional — things to avoid"
        />
        <Field label="Duration" value="8" unit="seconds" />
        <Field label="Frame rate" value="24" unit="fps" />
        <SegmentedControl
          label="Resolution"
          options={['480p', '720p', '1080p']}
          selected="720p"
        />
        <SegmentedControl
          label="Aspect"
          options={['16:9', '9:16', '1:1']}
          selected="16:9"
        />
        <ActionButton variant="primary">Queue video</ActionButton>
      </aside>
      <div className="workspace-results stack">
        <div className="row">
          <h2 className="eyebrow">Queue</h2>
          <span className="mono faint">2 pending · 1 done</span>
        </div>
        {videoJobs.map((job) => (
          <VideoJobCard job={job} key={job.id} />
        ))}
      </div>
    </div>
  )
}
