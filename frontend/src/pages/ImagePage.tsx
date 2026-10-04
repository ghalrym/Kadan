import {
  ActionButton,
  Field,
  ModeNavigation,
  SegmentedControl,
  UploadPlaceholder,
} from '../components/Controls'
import { MediaPlaceholder } from '../components/Media'
import { imageSets } from '../data/playground'

function ImageGallery() {
  return (
    <div className="workspace-results image-gallery">
      {imageSets.map((set) => (
        <section className="stack" key={set.id}>
          <header className="image-set-heading">
            <span className="badge type-Image">{set.mode}</span>
            <p>{set.prompt}</p>
            <span className="mono faint">{set.meta}</span>
          </header>
          <div className="image-grid">
            {set.seeds.map((seed) => (
              <MediaPlaceholder
                key={seed}
                aspect={set.aspect}
                label={`seed ${seed}`}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}

export default function ImagePage({ edit = false }: { edit?: boolean }) {
  return (
    <div className="workspace">
      <aside className="workspace-controls" aria-label="Image settings">
        <ModeNavigation
          label="Image mode"
          options={[
            { label: 'Generate from text', to: '/image' },
            { label: 'Edit an image', to: '/image/edit' },
          ]}
        />
        {edit && (
          <div className="field">
            <span className="eyebrow">Source image</span>
            <UploadPlaceholder
              title="Drop an image or click to upload"
              caption="PNG, JPG, WEBP · up to 20 MB"
            />
          </div>
        )}
        <Field
          label={edit ? 'Describe the edit' : 'Prompt'}
          multiline
          placeholder={
            edit
              ? 'e.g. Replace the background with a sunlit studio, keep the subject unchanged'
              : 'Describe the image you want…'
          }
        />
        <div className="image-options">
          <SegmentedControl
            label="Aspect"
            options={['1:1', '4:3', '3:4', '16:9']}
            selected="1:1"
          />
          <SegmentedControl
            label="Images"
            options={['1', '2', '4']}
            selected="4"
          />
        </div>
        {edit && (
          <label className="field">
            <span className="row">
              <span className="eyebrow">Edit strength</span>
              <span className="push-right mono muted">65%</span>
            </span>
            <input type="range" min={0} max={100} value={65} disabled />
          </label>
        )}
        <Field label="Seed" placeholder="Random" mono />
        <ActionButton variant="primary">
          {edit ? 'Apply edit' : 'Generate 4 images'}
        </ActionButton>
      </aside>
      <ImageGallery />
    </div>
  )
}
