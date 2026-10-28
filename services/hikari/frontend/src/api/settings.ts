import {
  cancelDownloadV1ModelsModelIdDownloadDelete,
  configureContextV1ModelsModelIdContextPut,
  downloadModelV1ModelsModelIdDownloadPost,
  getSettings,
} from './generated/sdk.gen'
import type { CheckpointStatus, SettingsResponse } from './generated/types.gen'

export type { CheckpointStatus, SettingsResponse }
export type DownloadState =
  | 'unassigned'
  | 'not_downloaded'
  | 'queued'
  | 'downloading'
  | 'complete'
  | 'failed'
export type ConfiguredModel = {
  id: string
  uses: string[]
  checkpoint: CheckpointStatus | null
}

const labels: Record<string, string> = {
  llm: 'Conversation',
  vision: 'Screenshot understanding',
  speech: 'Speech and calls',
  image: 'Image generation and editing',
  video: 'Video',
  transcription: 'Transcription',
  formatting: 'Transcript formatting',
  decisions: 'Decisions',
}

function settingsError(response: Response | undefined, error: unknown): Error {
  const detail = (error as { detail?: unknown } | undefined)?.detail
  return new Error(
    typeof detail === 'string'
      ? detail
      : response
        ? `Settings request failed (HTTP ${response.status}). Please retry.`
        : 'Cannot reach the settings API. Check that the backend is running, then retry.',
  )
}

/** The fixed stack can contain a shared model or a model with no connected worker. */
export function configuredModels(
  settings: SettingsResponse,
): ConfiguredModel[] {
  const models = new Map<string, ConfiguredModel>()
  for (const [feature, id] of Object.entries(settings.defaults)) {
    const row = models.get(id)
    if (row) row.uses.push(labels[feature] ?? feature)
    else
      models.set(id, {
        id,
        uses: [labels[feature] ?? feature],
        checkpoint: settings.models.find((model) => model.id === id) ?? null,
      })
  }
  return [...models.values()]
}

export async function fetchSettings(
  signal: AbortSignal,
): Promise<SettingsResponse> {
  const result = await getSettings({ signal })
  if (!result.response?.ok || !result.data)
    throw settingsError(result.response, result.error)
  return result.data
}

/** A caller must obtain acknowledgement of this checkpoint's own license first. */
export async function downloadModel(
  modelId: string,
  acknowledged: boolean,
  signal: AbortSignal,
): Promise<CheckpointStatus[]> {
  if (!acknowledged)
    throw new Error(
      'Review and acknowledge the model license before downloading.',
    )
  const result = await downloadModelV1ModelsModelIdDownloadPost({
    path: { model_id: modelId },
    body: { license_acknowledged: true },
    signal,
  })
  if (!result.response?.ok || !result.data)
    throw settingsError(result.response, result.error)
  return result.data.models
}

export async function cancelModelDownload(
  modelId: string,
  signal: AbortSignal,
): Promise<CheckpointStatus[]> {
  const result = await cancelDownloadV1ModelsModelIdDownloadDelete({
    path: { model_id: modelId },
    signal,
  })
  if (!result.response?.ok || !result.data)
    throw settingsError(result.response, result.error)
  return result.data.models
}

/** Null uses the model maximum; this changes the saved setting, not a running request. */
export async function configureModelContext(
  modelId: string,
  context: number | null,
  signal: AbortSignal,
): Promise<CheckpointStatus[]> {
  if (context !== null && (!Number.isSafeInteger(context) || context < 1))
    throw new Error(
      'Enter a positive whole-token context limit, or leave it blank to use the model maximum.',
    )
  const result = await configureContextV1ModelsModelIdContextPut({
    path: { model_id: modelId },
    body: { context_limit: context },
    signal,
  })
  if (!result.response?.ok || !result.data)
    throw settingsError(result.response, result.error)
  return result.data.models
}

export function downloadState(model: CheckpointStatus | null): DownloadState {
  if (!model) return 'unassigned'
  if (model.status === 'complete') return 'complete'
  if (model.status === 'failed' || model.status === 'cancelled') return 'failed'
  if (model.status === 'not_downloaded') return 'not_downloaded'
  return model.total_bytes > 0 ? 'downloading' : 'queued'
}
