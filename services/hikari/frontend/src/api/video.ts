import {
  cancelVideo,
  generateVideo,
  getSettings,
  getVideo,
  listVideos,
} from './generated/sdk.gen'
import type {
  VideoDefaults,
  VideoGenerationRequest,
  VideoJob,
} from './generated/types.gen'

export type { VideoJob, VideoGenerationRequest }

/**
 * Return whether server-reported job status still requires polling.
 */
export function isPendingVideo(job: VideoJob) {
  return job.status === 'Queued' || job.status === 'Rendering'
}

/**
 * Accept HTTP success or throw an actionable transport/provider/validation error.
 * A missing response represents a failed request, including an aborted fetch.
 */
function checkResponse(response?: Response) {
  if (response?.ok) return
  if (response?.status === 503)
    throw new Error(
      'Video generation is unavailable. Check the downloaded model and worker configuration.',
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

/**
 * Return job metadata after checking a truthy ID, string prompt, status and progress.
 * Throw for malformed metadata; a valid job does not establish a playable video URL.
 */
function validateJob(job: VideoJob | undefined): VideoJob {
  if (
    !job ||
    !job.id ||
    typeof job.prompt !== 'string' ||
    !['Queued', 'Rendering', 'Done', 'Failed', 'Cancelled'].includes(
      job.status,
    ) ||
    !Number.isFinite(job.progress) ||
    job.progress < 0 ||
    job.progress > 100
  )
    throw new Error('The API returned an invalid video job.')
  return job
}

/** Read the fixed stack's clip defaults from the API. */
export async function loadVideoDefaults(
  signal: AbortSignal,
): Promise<VideoDefaults> {
  const result = await getSettings({ signal })
  checkResponse(result.response)
  if (!result.data?.video_defaults)
    throw new Error('Video defaults are unavailable. Check Settings and retry.')
  return result.data.video_defaults
}

/**
 * Fetch and validate the queue, forwarding cancellation and rejecting invalid lists.
 */
export async function loadVideos(signal: AbortSignal) {
  const result = await listVideos({ signal })
  checkResponse(result.response)
  if (!Array.isArray(result.data?.jobs))
    throw new Error('The API returned an invalid video queue.')
  return result.data.jobs.map(validateJob)
}

/**
 * Fetch one cancellable job update and reject an ID mismatch or invalid metadata.
 */
export async function refreshVideo(id: string, signal: AbortSignal) {
  const result = await getVideo({ path: { video_id: id }, signal })
  checkResponse(result.response)
  const job = validateJob(result.data?.job)
  if (job.id !== id) throw new Error('The API returned a different video job.')
  return job
}

/**
 * Validate prompt, duration and frame rate before submitting typed settings.
 * Forward cancellation and return validated job metadata only on API success;
 * unavailable models or workers produce a visible error.
 */
export async function submitVideo(
  body: VideoGenerationRequest,
  signal: AbortSignal,
) {
  if (
    !body.prompt.trim() ||
    body.prompt.length > 8000 ||
    (body.negative_prompt?.length ?? 0) > 8000 ||
    (body.seconds != null &&
      (!Number.isInteger(body.seconds) ||
        body.seconds < 1 ||
        body.seconds > 15)) ||
    (body.fps !== undefined && body.fps !== 24)
  )
    throw new Error(
      'Enter a prompt and a whole-number duration from 1 to 15 seconds at 24 fps.',
    )
  const result = await generateVideo({
    body: { ...body, prompt: body.prompt.trim() },
    signal,
  })
  checkResponse(result.response)
  return validateJob(result.data?.job)
}

/** Stop the Hikari-owned worker; polling confirms cancellation after cleanup. */
export async function stopVideo(id: string) {
  const result = await cancelVideo({ path: { video_id: id } })
  checkResponse(result.response)
  return validateJob(result.data?.job)
}
