import type { InferenceProgress } from './generated/types.gen'

export type ImagePhase =
  | 'queued'
  | 'loading'
  | 'prompt'
  | 'denoise'
  | 'decode'

export type ImageProgressView = {
  jobId: string | null
  stale: boolean
  phase: ImagePhase
  /** Index of the user-facing step: queued, load model, generate. */
  milestone: 0 | 1 | 2
  title: string
  detail: string
  imageIndex: number
  imageCount: number
  step: number
  steps: number
  /** Share of the current image completed, or null outside generation. */
  imageFraction: number | null
  /** Share of the whole batch completed, or null while it cannot be estimated. */
  fraction: number | null
  observedNs: number | null
}

export const MILESTONES = ['Queued', 'Load model', 'Generate'] as const

function gigabytes(bytes: number) {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`
}

function loadingDetail(stage: string, value: number) {
  switch (stage) {
    case 'tokenizer_loaded':
      return 'Loading text encoder'
    case 'text_checkpoint_read_bytes':
      return `Reading text encoder · ${gigabytes(value)}`
    case 'text_loaded':
      return 'Loading denoiser'
    case 'denoiser_block_loaded':
      return `Loading denoiser · block ${value}`
    case 'denoiser_loaded':
      return 'Loading image decoder'
    case 'vae_checkpoint_read_bytes':
      return `Reading image decoder · ${gigabytes(value)}`
    case 'vae_loaded':
      return 'Model ready'
    default:
      return 'Loading model weights'
  }
}

/** Portion of one image finished at a phase; denoising dominates the runtime. */
function withinImage(phase: ImagePhase, step: number, steps: number) {
  if (phase === 'prompt') return 0.02
  if (phase === 'denoise') return 0.03 + (0.92 * step) / Math.max(steps, 1)
  if (phase === 'decode') return 0.96
  return 1
}

/** Retain only this request's progress across unavailable or delayed polls. */
export function describeImageProgress(
  progress: InferenceProgress | null | undefined,
  previous: ImageProgressView | null,
  requestId: string,
): ImageProgressView {
  if (previous?.jobId !== requestId) previous = null
  if (progress === undefined && previous) return { ...previous, stale: true }
  if (progress && (progress.job_id !== requestId || progress.workload !== 'image'))
    progress = undefined
  if (progress && previous?.observedNs != null && progress.observed_unix_ns < previous.observedNs)
    return previous
  const base = {
    jobId: requestId,
    stale: progress === undefined,
    imageIndex: previous?.imageIndex ?? 1,
    imageCount: previous?.imageCount ?? 1,
    step: previous?.step ?? 0,
    steps: previous?.steps ?? 0,
    observedNs: progress?.observed_unix_ns ?? null,
  }
  if (!progress) {
    if (previous) return { ...previous, stale: true }
    return {
      ...base,
      phase: 'queued',
      milestone: 0,
      title: 'Waiting for images',
      detail: 'Pending request',
      imageFraction: null,
      fraction: null,
    }
  }
  const image = progress.image
  const view = {
    ...base,
    imageIndex: image?.image_index ?? base.imageIndex,
    imageCount: image?.image_count ?? base.imageCount,
    step: image?.steps_completed ?? base.step,
    steps: image?.steps_total ?? base.steps,
  }
  const phase = image?.phase ?? 'queued'
  if (phase === 'queued' || phase === 'loading')
    return {
      ...view,
      phase,
      milestone: phase === 'queued' ? 0 : 1,
      title: phase === 'queued' ? 'Queued' : 'Loading model',
      detail:
        phase === 'loading'
          ? loadingDetail(progress.stage, progress.value)
          : progress.stage === 'waiting_for_resources'
            ? 'Waiting for GPU memory'
            : 'Preparing request',
      imageFraction: null,
      fraction: null,
    }
  const within = withinImage(phase, view.step, view.steps)
  return {
    ...view,
    phase,
    milestone: 2,
    title:
      view.imageCount > 1
        ? `Generating image ${view.imageIndex} of ${view.imageCount}`
        : 'Generating image',
    detail:
      phase === 'prompt'
        ? 'Reading the prompt'
        : phase === 'denoise'
          ? `Denoising · step ${Math.min(view.step + 1, view.steps)} of ${view.steps}`
          : 'Decoding pixels',
    imageFraction: within,
    fraction: (view.imageIndex - 1 + within) / Math.max(view.imageCount, 1),
  }
}

/** Extrapolate remaining seconds from the first and latest generation samples. */
export function estimateRemaining(
  start: { ns: number; fraction: number } | null,
  current: { ns: number; fraction: number } | null,
): number | null {
  if (!start || !current) return null
  const done = current.fraction - start.fraction
  const elapsed = (current.ns - start.ns) / 1e9
  if (done < 0.02 || elapsed <= 0) return null
  return Math.max(0, ((1 - current.fraction) * elapsed) / done)
}

/** Format seconds as `m:ss`, or `h:mm:ss` past an hour. */
export function formatDuration(seconds: number) {
  const total = Math.max(0, Math.round(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = String(total % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`
}
