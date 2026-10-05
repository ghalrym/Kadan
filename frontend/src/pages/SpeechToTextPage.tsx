import { ActionButton, Panel } from '../components/Controls'

export default function SpeechToTextPage() {
  return (
    <div className="scroll-page">
      <div className="recording-layout">
        <Panel className="recording-panel">
          <button
            type="button"
            className="record-button"
            disabled
            aria-label="Start recording"
          >
            <span />
          </button>
          <span className="recording-timer">0:00.0</span>
          <div className="row">
            <ActionButton>Discard</ActionButton>
            <ActionButton variant="muted">Submit</ActionButton>
          </div>
        </Panel>
      </div>
    </div>
  )
}
