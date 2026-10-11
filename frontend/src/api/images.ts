import { editImages, generateImages, listImages } from './generated/sdk.gen'
import type {
  ApiRoutesV1ImagesGenerationsImageRequest,
  ImageSet,
} from './generated/types.gen'

export type ImageOptions = ApiRoutesV1ImagesGenerationsImageRequest
/**
 * Send typed generation settings, or edit settings when source is supplied.
 * Forward cancellation and reject failures or incomplete PNG payloads.
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
  const image = response.data?.image
  if (
    !image || image.mime_type !== 'image/png' ||
    !Array.isArray(image.images_base64) ||
    image.images_base64.length !== (options.count ?? 4) ||
    !image.images_base64.every(value => typeof value === 'string' && value.trim().length > 0)
  )
    throw new Error('The image API returned incomplete PNG data. Please retry.')
  return image
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
