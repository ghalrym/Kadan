import { generateVideo, getVideo, listVideos } from './generated/sdk.gen'
import type { VideoGenerationRequest, VideoJob } from './generated/types.gen'

export type { VideoJob, VideoGenerationRequest }

export function isPendingVideo(job: VideoJob) {
  return job.status === 'Queued' || job.status === 'Rendering'
}

function checkResponse(response?: Response) {
  if (response?.ok) return
  if (response?.status === 503)
    throw new Error(
      'Video generation is unavailable: no provider is configured. No job was queued.',
    )
  if (response?.status === 404)
    throw new Error('This video job does not exist. Refresh the queue.')
  if (response?.status === 422)
    throw new Error(
      'The API rejected the video settings. Check the prompt and values.',
    )
  throw new Error(
    response
      ? `Video request failed (HTTP ${response.status}). Please retry.`
      : 'Cannot reach the video API. Check the backend and retry.',
  )
}

function validateJob(job: VideoJob | undefined): VideoJob {
  if (
    !job ||
    !job.id ||
    typeof job.prompt !== 'string' ||
    !['Queued', 'Rendering', 'Done'].includes(job.status) ||
    !Number.isFinite(job.progress) ||
    job.progress < 0 ||
    job.progress > 100
  )
    throw new Error('The API returned an invalid video job.')
  return job
}

export async function loadVideos(signal: AbortSignal) {
  const result = await listVideos({ signal })
  checkResponse(result.response)
  if (!Array.isArray(result.data?.jobs))
    throw new Error('The API returned an invalid video queue.')
  return result.data.jobs.map(validateJob)
}

export async function refreshVideo(id: string, signal: AbortSignal) {
  const result = await getVideo({ path: { video_id: id }, signal })
  checkResponse(result.response)
  const job = validateJob(result.data?.job)
  if (job.id !== id) throw new Error('The API returned a different video job.')
  return job
}

export async function submitVideo(
  body: VideoGenerationRequest,
  signal: AbortSignal,
) {
  if (
    !body.prompt.trim() ||
    body.prompt.length > 8000 ||
    (body.negative_prompt?.length ?? 0) > 8000 ||
    !Number.isInteger(body.duration) ||
    body.duration! < 1 ||
    body.duration! > 120 ||
    !Number.isInteger(body.fps) ||
    body.fps! < 1 ||
    body.fps! > 120
  )
    throw new Error(
      'Enter a prompt and whole-number duration and frame rate between 1 and 120.',
    )
  const result = await generateVideo({
    body: { ...body, prompt: body.prompt.trim() },
    signal,
  })
  checkResponse(result.response)
  return validateJob(result.data?.job)
}
