import type { ModelStatus } from '../api/generated'
import { ModelPicker } from './ModelPicker'

/** Download the one formatting checkpoint through the shared model manager. */
export function FormattingModel({ model, pending, downloading, download, cancel }: { model?: ModelStatus; pending: boolean; downloading: boolean; download: (model: ModelStatus) => void; cancel: (model: ModelStatus) => void }) {
  const active = model?.status === 'downloading' || model?.status === 'cancelling'
  const failed = model?.status === 'failed' || model?.status === 'cancelled'
  return <div className="stack compact"><ModelPicker pickerId="Formatting" label="Formatting model" models={model ? [model] : []} current={model} selectedId={model?.status === 'complete' ? model.id : null} pending={pending} downloading={downloading} choose={() => {}} download={download} />
    {model && (active || failed) && <div className="model-download-status"><div className="model-download-heading"><p role="status">{model.status}</p><button type="button" className="button" disabled={pending || model.status === 'cancelling' || (!active && downloading)} onClick={() => active ? cancel(model) : download(model)}>{active ? 'Cancel download' : 'Retry download'}</button></div>
      {active && <progress className="progress" aria-label="S1-mini download progress" max={model.total_bytes || 1} value={model.total_bytes ? model.downloaded_bytes : undefined} />}
      {model.error && <p role="alert" className="error">{model.error}</p>}
    </div>}
  </div>
}
