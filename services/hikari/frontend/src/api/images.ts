import { editImages, generateImages, listImages } from './generated/sdk.gen'
import type {
  ApiRoutesV1ImagesGenerationsImageRequest,
  ImageSet,
} from './generated/types.gen'

export type ImageOptions = ApiRoutesV1ImagesGenerationsImageRequest

/** Random job identity also works on the home server's HTTP origin. */
export function createImageRequestId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('')
}

/**
 * Send typed generation settings, or edit settings when source is supplied.
 * Forward cancellation and reject failures or incomplete PNG payloads.
 */
export async function requestImages(
  options: ImageOptions,
  signal: AbortSignal,
  source?: { image: string; mask?: string | null; strength: number },
  requestId?: string,
): Promise<ImageSet> {
  const headers = requestId ? { 'X-Kadan-Request-Id': requestId } : undefined
  const response = source
    ? await editImages({ body: { ...options, ...source }, signal, headers })
    : await generateImages({ body: options, signal, headers })
  if (!response.response)
    throw new Error(
      'Cannot reach the image API. Check that the backend is running, then retry.',
    )
  if (!response.response.ok) {
    const detail = (response.error as { detail?: unknown } | undefined)?.detail
    if (typeof detail === 'string' && detail.trim()) throw new Error(detail)
    if (Array.isArray(detail)) {
      const messages = detail.flatMap((issue: { loc?: unknown[]; msg?: unknown }) =>
        typeof issue?.msg === 'string'
          ? [`${Array.isArray(issue.loc) ? issue.loc.filter(part => part !== 'body').join('.') + ': ' : ''}${issue.msg}`]
          : [],
      )
      if (messages.length) throw new Error(messages.join(' '))
    }
    throw new Error(
      `Image request failed (HTTP ${response.response.status}). Check your inputs and retry.`,
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
