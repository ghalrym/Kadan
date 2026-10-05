import { editImages, generateImages, listImages } from './generated/sdk.gen'
import type {
  ApiRoutesV1ImagesGenerationsImageRequest,
  ImageSet,
} from './generated/types.gen'

export type ImageOptions = ApiRoutesV1ImagesGenerationsImageRequest
/**
 * Send typed generation settings, or edit settings when source is supplied.
 * Forward the abort signal and reject network/HTTP failures. Require deliverable image URLs before accepting a successful response.
 */
export async function requestImages(
  options: ImageOptions,
  signal: AbortSignal,
  source?: { image: string },
): Promise<ImageSet> {
  const response = source
    ? await editImages({ body: { ...options, ...source }, signal })
    : await generateImages({ body: options, signal })
  if (!response.response)
    throw new Error(
      'Cannot reach the image API. Check that the backend is running, then retry.',
    )
  if (!response.response.ok) {
    const error = response.error as { detail?: unknown } | undefined
    throw new Error(typeof error?.detail === 'string' ? error.detail : `Image request failed (HTTP ${response.response.status}).`)
  }
  const result = response.data?.image
  if (!result?.urls?.length || (result.seeds.length !== 0 && result.urls.length !== result.seeds.length))
    throw new Error('The API returned metadata without image files. No generated image is available.')
  return result
}
/**
 * Fetch the image metadata list with caller-controlled cancellation.
 * Reject unavailable or non-array history; entries are not playable image files.
 */
export async function imageHistory(signal: AbortSignal): Promise<ImageSet[]> {
  const result = await listImages({ signal })
  if (!result.response?.ok || !Array.isArray(result.data?.images))
    throw new Error(
      'Could not load image history. Check the API connection and retry.',
    )
  return result.data.images
}
