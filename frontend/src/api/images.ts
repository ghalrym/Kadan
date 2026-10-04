import { editImages, generateImages, listImages } from './generated/sdk.gen'
import type {
  ApiRoutesV1ImagesGenerationsImageRequest,
  ImageSet,
} from './generated/types.gen'

export type ImageOptions = ApiRoutesV1ImagesGenerationsImageRequest
/**
 * Send typed generation settings, or edit settings when source is supplied.
 * Forward the abort signal and reject network/HTTP failures. Even a 2xx metadata
 * response is rejected because this contract has no deliverable image URLs.
 */
export async function requestImages(
  options: ImageOptions,
  signal: AbortSignal,
  source?: { image: string; strength: number },
): Promise<ImageSet> {
  const response = source
    ? await editImages({ body: { ...options, ...source }, signal })
    : await generateImages({ body: options, signal })
  if (!response.response)
    throw new Error(
      'Cannot reach the image API. Check that the backend is running, then retry.',
    )
  if (!response.response.ok) {
    if (response.response?.status === 503)
      throw new Error(
        'No image provider is configured. Generation and editing are unavailable.',
      )
    throw new Error(
      `Image request failed (HTTP ${response.response?.status}). Check your inputs and retry.`,
    )
  }
  // The current API has metadata only, with no deliverable image URLs. Never
  // turn a successful-looking placeholder into a fabricated generated image.
  throw new Error(
    'The API returned metadata without image files. No generated image is available.',
  )
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
