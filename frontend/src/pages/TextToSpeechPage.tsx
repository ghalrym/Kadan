import {
  ActionButton,
  Field,
  ModeNavigation,
  UploadPlaceholder,
} from '../components/Controls'
import { AudioCard } from '../components/Media'
import { speechScript, voiceDescription } from '../data/playground'

export default function TextToSpeechPage({
  clone = false,
}: {
  clone?: boolean
}) {
  return (
    <div className="workspace workspace--speech">
      <aside className="workspace-controls" aria-label="Speech settings">
        <div className="field">
          <span className="eyebrow">Voice</span>
          <ModeNavigation
            label="Voice mode"
            options={[
              { label: 'Describe a voice', to: '/tts' },
              { label: 'Clone a voice', to: '/tts/clone' },
            ]}
          />
        </div>
        {clone ? (
          <UploadPlaceholder
            title="Upload a voice sample"
            caption="WAV or MP3 · 10–30 s of clean speech"
          />
        ) : (
          <textarea
            className="input"
            aria-label="Voice description"
            rows={3}
            readOnly
            value={voiceDescription}
          />
        )}
        <div className="speech-script">
          <Field label="Script" value={speechScript} multiline rows={6} />
          <span className="mono faint character-count">
            {speechScript.length} / 5,000
          </span>
        </div>
        <ActionButton variant="primary">Generate speech</ActionButton>
      </aside>
      <div className="workspace-results stack">
        <h2 className="eyebrow">Generated audio</h2>
        <AudioCard
          voice="Described · warm, low, calm"
          meta="F5-TTS · 3.6 s · WAV · 14:11"
          script="Your order has shipped and will arrive on Thursday."
          time="0:03"
        />
      </div>
    </div>
  )
}
