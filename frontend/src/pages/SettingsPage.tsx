import { Panel, SectionHeading } from '../components/Controls'
import { modelSettings } from '../data/playground'

function ModelIcon({ type }: { type: string }) {
  return (
    <svg
      className={`model-icon type-${type}`}
      width="13"
      height="13"
      viewBox="0 0 12 12"
      aria-hidden="true"
    >
      <circle cx="6" cy="3" r="2.5" />
      <circle cx="8.85" cy="5.07" r="2.5" />
      <circle cx="7.76" cy="8.43" r="2.5" />
      <circle cx="4.24" cy="8.43" r="2.5" />
      <circle cx="3.15" cy="5.07" r="2.5" />
      <circle className="model-icon-center" cx="6" cy="6" r="1.5" />
    </svg>
  )
}

export default function SettingsPage() {
  return (
    <div className="scroll-page">
      <div className="settings-layout stack">
        <SectionHeading>Models</SectionHeading>
        <Panel className="model-settings">
          {modelSettings.map((model) => (
            <div className="model-row" key={model.type}>
              <label htmlFor={`model-${model.type}`}>
                <ModelIcon type={model.type} />
                {model.label}
              </label>
              <select
                className="input"
                id={`model-${model.type}`}
                value={model.selected}
                disabled
              >
                {model.options.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </div>
          ))}
          <div className="model-row">
            <span className="model-label">
              <ModelIcon type="STT" />
              Whisper S1 Mini formatting
            </span>
            <button
              type="button"
              role="switch"
              aria-checked="true"
              aria-label="Whisper S1 Mini formatting"
              disabled
              className="switch"
            >
              <span />
            </button>
          </div>
        </Panel>
      </div>
    </div>
  )
}
