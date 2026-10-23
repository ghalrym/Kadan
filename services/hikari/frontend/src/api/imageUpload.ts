const MAX_BYTES = 20 * 1024 * 1024
const MAX_SIDE = 3072
const MAX_PIXELS = 4608 * 1024
const MASK_TILE_SIDE = 512
const IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp'])

type Dimensions = { width: number; height: number }
export type ImageUpload = Dimensions & {
  name: string
  dataUrl: string
  originalWidth: number
  originalHeight: number
}

export function validateImageFile(file: Pick<File, 'size' | 'type'>) {
  if (!IMAGE_TYPES.has(file.type)) throw new Error('Choose a PNG, JPEG, or WEBP image.')
  if (file.size === 0) throw new Error('This image is empty. Choose another file.')
  if (file.size > MAX_BYTES) throw new Error('Choose an image no larger than 20 MiB.')
}

/** Fit within the worker limits, then align the canvas to its latent grid. */
export function imageUploadDimensions(width: number, height: number): Dimensions {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0)
    throw new Error('This image has invalid dimensions. Choose another file.')
  const scale = Math.min(1, MAX_SIDE / width, MAX_SIDE / height, Math.sqrt(MAX_PIXELS / (width * height)))
  return {
    width: Math.max(32, Math.floor(width * scale / 32) * 32),
    height: Math.max(32, Math.floor(height * scale / 32) * 32),
  }
}

/** Decode locally; only the normalized PNG leaves the browser on submission. */
export async function prepareImageUpload(file: File, source?: ImageUpload): Promise<ImageUpload> {
  validateImageFile(file)
  const url = URL.createObjectURL(file)
  try {
    const image = new Image()
    image.src = url
    try {
      await image.decode()
    } catch {
      throw new Error('This image could not be read. Choose a valid PNG, JPEG, or WEBP file.')
    }
    const originalWidth = image.naturalWidth
    const originalHeight = image.naturalHeight
    if (source && (originalWidth !== source.originalWidth || originalHeight !== source.originalHeight))
      throw new Error(`The mask must match the original image: ${source.originalWidth} × ${source.originalHeight} pixels.`)
    const { width, height } = source ?? imageUploadDimensions(originalWidth, originalHeight)
    const canvas = document.createElement('canvas')
    canvas.width = source ? Math.min(MASK_TILE_SIDE, originalWidth) : width
    canvas.height = source ? Math.min(MASK_TILE_SIDE, originalHeight) : height
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Image preparation is unavailable in this browser.')
    if (source) {
      // Resampling can round original transparency to opaque, so inspect native pixels first.
      for (let y = 0; y < originalHeight; y += MASK_TILE_SIDE) {
        for (let x = 0; x < originalWidth; x += MASK_TILE_SIDE) {
          const tileWidth = Math.min(MASK_TILE_SIDE, originalWidth - x)
          const tileHeight = Math.min(MASK_TILE_SIDE, originalHeight - y)
          context.clearRect(0, 0, tileWidth, tileHeight)
          context.drawImage(image, x, y, tileWidth, tileHeight, 0, 0, tileWidth, tileHeight)
          const pixels = context.getImageData(0, 0, tileWidth, tileHeight)
          for (let index = 3; index < pixels.data.length; index += 4) {
            if (pixels.data[index] < 255)
              throw new Error('Use an opaque mask: black preserves the image, white edits it. Transparent masks are not supported.')
          }
        }
      }
      canvas.width = width
      canvas.height = height
    }
    // Source and mask use the same resize to the worker's 32-pixel grid.
    context.drawImage(image, 0, 0, width, height)
    if (source) {
      const pixels = context.getImageData(0, 0, width, height)
      for (let index = 0; index < pixels.data.length; index += 4) {
        const value = Math.round(0.2126 * pixels.data[index] + 0.7152 * pixels.data[index + 1] + 0.0722 * pixels.data[index + 2])
        pixels.data[index] = value
        pixels.data[index + 1] = value
        pixels.data[index + 2] = value
      }
      context.putImageData(pixels, 0, 0)
    }
    const dataUrl = canvas.toDataURL('image/png')
    if (!dataUrl.startsWith('data:image/png;base64,')) throw new Error('Could not prepare this image. Choose a smaller file.')
    if (dataUrl.length > 22 + 4 * Math.ceil(MAX_BYTES / 3))
      throw new Error('The prepared PNG exceeds 20 MiB. Choose a smaller image.')
    return { name: file.name, dataUrl, width, height, originalWidth, originalHeight }
  } finally {
    URL.revokeObjectURL(url)
  }
}
