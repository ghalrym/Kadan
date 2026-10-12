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

// Worker stage names arrive as `phase || state`; most are completion events.
const STAGES: Record<string, ImagePhase> = {
  waiting_for_resources: 'queued',
  preparing: 'queued',
  loading: 'loading',
  checkpoint_tensor_bytes: 'loading',
  expanded_f32_inventory_bytes: 'loading',
  nonfloating_inventory_bytes: 'loading',
  tokenizer_loaded: 'loading',
  text_checkpoint_read_bytes: 'loading',
  text_loaded: 'loading',
  denoiser_block_loaded: 'loading',
  denoiser_loaded: 'loading',
  vae_checkpoint_read_bytes: 'loading',
  vae_loaded: 'loading',
  running: 'prompt',
  image_started: 'prompt',
  text_layer: 'prompt',
  conditioning: 'denoise',
  block: 'denoise',
  step_completed: 'denoise',
  vae_up_block: 'decode',
}

const WORKLOADS: Record<string, string> = {
  llm: 'a chat',
  video: 'a video',
  stt: 'a transcription',
  tts: 'a speech',
  decisions: 'a decision',
  models: 'a model download',
}

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

/**
 * Describe shared worker activity, never the browser request's progress.
 * Undefined means no successful poll; null is an authoritative idle snapshot.
 * Only carry ambiguous phases and counters forward within the same worker job.
 */
export function describeImageProgress(
  progress: InferenceProgress | null | undefined,
  previous: ImageProgressView | null,
): ImageProgressView {
  if (progress === undefined && previous) return { ...previous, stale: true }
  if (!progress || previous?.jobId !== progress.job_id) previous = null
  const base = {
    jobId: progress?.job_id ?? null,
    stale: progress === undefined,
    imageIndex: previous?.imageIndex ?? 1,
    imageCount: previous?.imageCount ?? 1,
    step: previous?.step ?? 0,
    steps: previous?.steps ?? 0,
    observedNs: progress?.observed_unix_ns ?? null,
  }
  if (!progress || progress.workload !== 'image') {
    return {
      ...base,
      phase: 'queued',
      milestone: 0,
      title: progress ? 'Worker busy' : progress === undefined ? 'Worker status unavailable' : 'Worker idle',
      detail: progress
        ? `Processing ${WORKLOADS[progress.workload] ?? 'another'} job`
        : 'No active worker job reported',
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
  const phase =
    STAGES[progress.stage] ??
    previous?.phase ??
    (view.steps && view.step >= view.steps ? 'decode' : 'loading')
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
          ? `Denoising · step ${view.step} of ${view.steps}`
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
