import { cancelVideo, generateVideo, getVideo, listVideos } from './generated/sdk.gen'
import type { VideoGenerationRequest, VideoJob } from './generated/types.gen'

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
    !['Queued', 'Rendering', 'Done', 'Failed', 'Cancelled'].includes(job.status) ||
    !Number.isFinite(job.progress) ||
    job.progress < 0 ||
    job.progress > 100
  )
    throw new Error('The API returned an invalid video job.')
  return job
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

/** Stop the Kadan-owned worker; polling confirms cancellation after cleanup. */
export async function stopVideo(id: string) {
  const result = await cancelVideo({ path: { video_id: id } })
  checkResponse(result.response)
  return validateJob(result.data?.job)
}

/** Upload conditioning bytes and return their opaque Kadan input identifier. */
export async function uploadVideoConditioning(file: File, kind: 'image' | 'video', signal: AbortSignal): Promise<string> {
  const response = await fetch(`/v1/videos/inputs?kind=${kind}`, {
    method: 'POST', headers: { 'Content-Type': file.type }, body: file, signal,
  })
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Conditioning upload failed.')
  if (typeof data.id !== 'string' || !/^[0-9a-f]{32}$/.test(data.id)) throw new Error('Conditioning upload returned an invalid ID.')
  return data.id
}
