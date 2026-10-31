import {
  downloadModelV1ModelsModelIdDownloadPost,
  listModelsV1ModelsGet,
} from './generated/sdk.gen'
import type { CheckpointStatus } from './generated/types.gen'
import { capabilities, type Capability } from '../data/models'

export type { CheckpointStatus }
export type DownloadState =
  | 'unavailable'
  | 'not_downloaded'
  | 'queued'
  | 'downloading'
  | 'complete'
  | 'failed'

/** One capability and its checkpoint; null while the worker does not report it. */
export interface CapabilityModel {
  capability: Capability
  model: CheckpointStatus | null
}

/**
 * Turn a missing or rejected response into an actionable Settings error,
 * preferring the API's own detail.
 */
function settingsError(response: Response | undefined, error: unknown): Error {
  const detail = (error as { detail?: unknown } | undefined)?.detail
  if (typeof detail === 'string') return new Error(detail)
  return new Error(
    response
      ? `Settings request failed (HTTP ${response.status}). Please retry.`
      : 'Cannot reach the settings API. Check that the backend is running, then retry.',
  )
}

/** Pair each capability with its checkpoint from a model snapshot. */
export function capabilityModels(models: CheckpointStatus[]): CapabilityModel[] {
  return capabilities.map((capability) => ({
    capability,
    model: models.find((model) => model.id === capability.id) ?? null,
  }))
}

/** Read each capability's checkpoint and download state. */
export async function fetchModels(signal: AbortSignal): Promise<CapabilityModel[]> {
  const result = await listModelsV1ModelsGet({ signal })
  if (!result.response?.ok || !result.data) throw settingsError(result.response, result.error)
  return capabilityModels(result.data.models)
}

/**
 * Queue every incomplete, idle checkpoint in order. The caller has shown the
 * license notices and obtained explicit acknowledgement.
 */
export async function downloadModels(
  rows: CapabilityModel[],
  acknowledged: boolean,
  signal: AbortSignal,
): Promise<CapabilityModel[]> {
  if (!acknowledged) throw new Error('Review and acknowledge the model licenses before downloading.')
  let current = rows
  for (const { model } of rows) {
    if (!model || !canDownload(model)) continue
    const result = await downloadModelV1ModelsModelIdDownloadPost({
      path: { model_id: model.id }, body: { license_acknowledged: true }, signal,
    })
    if (!result.response?.ok || !result.data) throw settingsError(result.response, result.error)
    current = capabilityModels(result.data.models)
  }
  return current
}

/**
 * A queued download has no observed size yet; cancelling still occupies the
 * queue until the worker confirms.
 */
export function downloadState(model: CheckpointStatus | null): DownloadState {
  if (!model) return 'unavailable'
  if (model.status === 'complete') return 'complete'
  if (model.status === 'failed' || model.status === 'cancelled') return 'failed'
  if (model.status === 'not_downloaded') return 'not_downloaded'
  return model.total_bytes > 0 ? 'downloading' : 'queued'
}

function canDownload(model: CheckpointStatus): boolean {
  const state = downloadState(model)
  return state === 'not_downloaded' || state === 'failed'
}

/** Reported checkpoints, deduplicated when capabilities share one. */
function reported(rows: CapabilityModel[]): CheckpointStatus[] {
  const models = new Map<string, CheckpointStatus>()
  for (const { model } of rows) if (model) models.set(model.id, model)
  return [...models.values()]
}

export function modelsDownloading(rows: CapabilityModel[]): boolean {
  return reported(rows).some((model) => {
    const state = downloadState(model)
    return state === 'queued' || state === 'downloading'
  })
}

export function modelsCanDownload(rows: CapabilityModel[]): boolean {
  return reported(rows).some(canDownload)
}

export function modelsComplete(rows: CapabilityModel[]): boolean {
  return rows.every(({ model }) => model?.status === 'complete')
}

/**
 * Overall progress in percent, weighting each checkpoint by its catalog
 * estimate until the worker reports the exact size.
 */
export function modelsProgress(rows: CapabilityModel[]): number {
  let done = 0
  let total = 0
  for (const model of reported(rows)) {
    const size = model.total_bytes || model.estimated_bytes
    total += size
    done += model.status === 'complete' ? size : Math.min(model.downloaded_bytes, size)
  }
  return total ? Math.floor((done / total) * 100) : 0
}

/** License notices the user must read before downloads start. */
export function licenseNotices(rows: CapabilityModel[]): CheckpointStatus[] {
  return reported(rows).filter((model) => model.license_notice && model.status !== 'complete')
}
