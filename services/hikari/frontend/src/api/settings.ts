import {
  downloadModelV1ModelsModelIdDownloadPost,
  getSettings,
  listModelsV1ModelsGet,
  updateSettings,
} from './generated/sdk.gen'
import type {
  CheckpointStatus,
  SettingsResponse,
  TierModel,
  TierSetting,
} from './generated/types.gen'

export type { CheckpointStatus, SettingsResponse, TierModel, TierSetting }
export type TierId = TierSetting['id']
export type DownloadState =
  | 'unassigned'
  | 'not_downloaded'
  | 'queued'
  | 'downloading'
  | 'complete'
  | 'failed'

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

/** Read the tiers, the selected tier and each assigned checkpoint's download state. */
export async function fetchSettings(signal: AbortSignal): Promise<SettingsResponse> {
  const result = await getSettings({ signal })
  if (!result.response?.ok || !result.data) throw settingsError(result.response, result.error)
  return result.data
}

/**
 * Save the tier and queue its incomplete downloads. The caller has shown the
 * tier's license notices before acknowledging them.
 */
export async function chooseTier(
  tier: TierId,
  signal: AbortSignal,
): Promise<SettingsResponse> {
  const result = await updateSettings({
    body: { tier, license_acknowledged: true },
    signal,
  })
  if (!result.response?.ok || !result.data) throw settingsError(result.response, result.error)
  return result.data
}

export function experimentalModels(models: CheckpointStatus[]): CheckpointStatus[] {
  return models.filter((model) => model.id.startsWith('lfm3-'))
}

export async function fetchExperimentalModels(signal: AbortSignal): Promise<CheckpointStatus[]> {
  const result = await listModelsV1ModelsGet({ signal })
  if (!result.response?.ok || !result.data) throw settingsError(result.response, result.error)
  return experimentalModels(result.data.models)
}

/** The dialog must obtain explicit acknowledgement before queuing any download. */
export async function downloadExperimentalModel(
  modelId: string,
  acknowledged: boolean,
  signal: AbortSignal,
): Promise<CheckpointStatus[]> {
  if (!acknowledged) throw new Error('Review and acknowledge the model license before downloading.')
  const result = await downloadModelV1ModelsModelIdDownloadPost({
    path: { model_id: modelId }, body: { license_acknowledged: true }, signal,
  })
  if (!result.response?.ok || !result.data) throw settingsError(result.response, result.error)
  return experimentalModels(result.data.models)
}

/**
 * A queued download has no observed size yet; cancelling still occupies the
 * queue until the worker confirms.
 */
export function downloadState(model: CheckpointStatus | null): DownloadState {
  if (!model) return 'unassigned'
  if (model.status === 'complete') return 'complete'
  if (model.status === 'failed' || model.status === 'cancelled') return 'failed'
  if (model.status === 'not_downloaded') return 'not_downloaded'
  return model.total_bytes > 0 ? 'downloading' : 'queued'
}

/** Assigned checkpoints of a tier, deduplicated when modalities share one. */
function assigned(tier: TierSetting): CheckpointStatus[] {
  const models = new Map<string, CheckpointStatus>()
  for (const { model } of tier.models) if (model) models.set(model.id, model)
  return [...models.values()]
}

export function tierDownloading(tier: TierSetting): boolean {
  return assigned(tier).some((model) => {
    const state = downloadState(model)
    return state === 'queued' || state === 'downloading'
  })
}

export function tierCanDownload(tier: TierSetting): boolean {
  return !tierDownloading(tier) && !tierComplete(tier)
}

export function tierComplete(tier: TierSetting): boolean {
  return assigned(tier).every((model) => model.status === 'complete')
}

/**
 * Whole-tier progress in percent, weighting each checkpoint by its catalog
 * estimate until the worker reports the exact size.
 */
export function tierProgress(tier: TierSetting): number {
  let done = 0
  let total = 0
  for (const model of assigned(tier)) {
    const size = model.total_bytes || model.estimated_bytes
    total += size
    done += model.status === 'complete' ? size : Math.min(model.downloaded_bytes, size)
  }
  return total ? Math.floor((done / total) * 100) : 0
}

/** License notices the user must read before the tier's downloads start. */
export function licenseNotices(tier: TierSetting): CheckpointStatus[] {
  return assigned(tier).filter(
    (model) => model.license_notice && model.status !== 'complete',
  )
}
