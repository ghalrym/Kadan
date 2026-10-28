import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { SourceTextModule, SyntheticModule } from 'node:vm'

// Replace only the generated HTTP boundary; exercise the actual API adapter.
async function imageApi(response) {
  const calls = []
  const sdk = new SyntheticModule(['generateImages', 'editImages', 'listImages'], function () {
    for (const name of ['generateImages', 'editImages', 'listImages'])
      this.setExport(name, async args => { calls.push({ name, args }); return response })
  })
  const source = await readFile(new URL('../src/api/images.ts', import.meta.url), 'utf8')
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link(specifier => {
    assert.equal(specifier, './generated/sdk.gen')
    return sdk
  })
  await module.evaluate()
  return { api: module.namespace, calls }
}

const image = {
  id: 'job', mode: 'Generate', prompt: 'A red mug', aspect: 'square',
  seeds: [42], meta: 'PNG', images_base64: ['cG5n'], mime_type: 'image/png',
}

test('returns delivered PNG data and forwards settings and cancellation', async () => {
  const { api, calls } = await imageApi({ response: { ok: true }, data: { image } })
  const options = { prompt: 'A red mug', count: 1 }
  const signal = new AbortController().signal
  assert.equal(await api.requestImages(options, signal), image)
  assert.equal(calls[0].name, 'generateImages')
  assert.equal(calls[0].args.body, options)
  assert.equal(calls[0].args.signal, signal)
})

test('keeps all images in a multi-image result', async () => {
  const batch = { ...image, seeds: [42, 43], images_base64: ['cG5n', 'cG5n'] }
  const { api } = await imageApi({ response: { ok: true }, data: { image: batch } })
  assert.equal(await api.requestImages({ prompt: 'mugs', count: 2 }, new AbortController().signal), batch)
})

test('rejects metadata-only, missing, mismatched and invalid PNG payloads', async () => {
  for (const value of [undefined, { ...image, images_base64: undefined },
    { ...image, images_base64: [] }, { ...image, images_base64: [''] },
    { ...image, images_base64: [null] }, { ...image, images_base64: 'cG5n' },
    { ...image, images_base64: ['cG5n', 'cG5n'] }, { ...image, mime_type: 'text/html' }]) {
    const { api } = await imageApi({ response: { ok: true }, data: { image: value } })
    await assert.rejects(api.requestImages({ prompt: 'mug', count: 1 }, new AbortController().signal), /incomplete PNG data/)
  }
})

test('preserves transport and provider failures', async () => {
  for (const [response, message] of [[{}, /Cannot reach/],
    [{ response: { ok: false, status: 503 } }, /HTTP 503/],
    [{ response: { ok: false, status: 422 } }, /HTTP 422/]]) {
    const { api } = await imageApi(response)
    await assert.rejects(api.requestImages({ prompt: 'mug', count: 1 }, new AbortController().signal), message)
  }
})


test('sends the exact request ID header without changing generation settings', async () => {
  const { api, calls } = await imageApi({ response: { ok: true }, data: { image } })
  const options = { prompt: 'mug', count: 1 }
  await api.requestImages(options, new AbortController().signal, undefined, 'a'.repeat(32))
  assert.deepEqual(calls[0].args.headers, { 'X-Kadan-Request-Id': 'a'.repeat(32) })
  assert.equal(calls[0].args.body, options)
})


test('creates a native job ID when randomUUID is unavailable on HTTP', async () => {
  const { api } = await imageApi({})
  const original = Object.getOwnPropertyDescriptor(globalThis, 'crypto')
  let bytesRequested = 0
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: {
    getRandomValues(bytes) {
      bytesRequested = bytes.length
      return bytes.map((_, index) => index)
    },
  } })
  try {
    assert.equal(api.createImageRequestId(), '000102030405060708090a0b0c0d0e0f')
    assert.equal(bytesRequested, 16)
  } finally {
    Object.defineProperty(globalThis, 'crypto', original)
  }
})

test('sends edit image, mask, zero strength, cancellation and the same request ID', async () => {
  const { api, calls } = await imageApi({ response: { ok: true }, data: { image } })
  const options = { prompt: 'A blue mug', count: 1, seed: 42 }
  const source = { image: 'data:image/png;base64,c291cmNl', mask: 'data:image/png;base64,bWFzaw==', strength: 0 }
  const signal = new AbortController().signal
  assert.equal(await api.requestImages(options, signal, source, 'e'.repeat(32)), image)
  assert.equal(calls[0].name, 'editImages')
  assert.deepEqual(calls[0].args.body, { ...options, ...source })
  assert.equal(calls[0].args.signal, signal)
  assert.deepEqual(calls[0].args.headers, { 'X-Kadan-Request-Id': 'e'.repeat(32) })
})

test('shows backend edit errors and field validation details', async () => {
  for (const [detail, message] of [
    ['The selected model does not support editing.', /selected model/],
    [[{ loc: ['body', 'mask'], msg: 'Mask dimensions must match the source' }], /mask: Mask dimensions/],
  ]) {
    const { api } = await imageApi({ response: { ok: false, status: 422 }, error: { detail } })
    await assert.rejects(api.requestImages({ prompt: 'mug', count: 1 }, new AbortController().signal), message)
  }
})

async function uploadApi() {
  const source = await readFile(new URL('../src/api/imageUpload.ts', import.meta.url), 'utf8')
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link(() => { throw new Error('Unexpected upload dependency') })
  await module.evaluate()
  return module.namespace
}

test('validates upload type, emptiness and the inclusive 20 MiB limit', async () => {
  const { validateImageFile } = await uploadApi()
  for (const type of ['image/png', 'image/jpeg', 'image/webp'])
    assert.doesNotThrow(() => validateImageFile({ type, size: 20 * 1024 * 1024 }))
  for (const [file, message] of [
    [{ type: 'image/gif', size: 100 }, /PNG, JPEG, or WEBP/],
    [{ type: 'image/png', size: 0 }, /empty/],
    [{ type: 'image/png', size: 20 * 1024 * 1024 + 1 }, /20 MiB/],
  ]) assert.throws(() => validateImageFile(file), message)
})

test('normalizes dimensions to the native grid and resource limits', async () => {
  const { imageUploadDimensions } = await uploadApi()
  assert.deepEqual(imageUploadDimensions(1024, 768), { width: 1024, height: 768 })
  assert.deepEqual(imageUploadDimensions(4032, 3024), { width: 2496, height: 1856 })
  for (const [width, height] of [[1, 1], [1, 99999], [99999, 1], [99999, 99999], [3840, 2160]]) {
    const output = imageUploadDimensions(width, height)
    assert.ok(output.width >= 32 && output.height >= 32)
    assert.ok(output.width <= 3072 && output.height <= 3072)
    assert.ok(output.width * output.height <= 4608 * 1024)
    assert.equal(output.width % 32, 0)
    assert.equal(output.height % 32, 0)
  }
  assert.throws(() => imageUploadDimensions(0, 100), /invalid dimensions/)
})

// Replace browser decoding/canvas boundaries only; exercise normalization and validation.
async function withUploadBrowser({ width = 100, height = 70, pixels = [255, 255, 255, 255], originalPixels = [], decodeFails = false } = {}, run) {
  const calls = { revoked: [], draws: [], reads: [], encodes: 0, outputPixels: null }
  const original = Object.fromEntries(['Image', 'document'].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]))
  const create = URL.createObjectURL
  const revoke = URL.revokeObjectURL
  URL.createObjectURL = () => 'blob:local-upload'
  URL.revokeObjectURL = url => calls.revoked.push(url)
  Object.defineProperty(globalThis, 'Image', { configurable: true, value: class {
    naturalWidth = width
    naturalHeight = height
    async decode() { if (decodeFails) throw new Error('invalid image') }
  } })
  Object.defineProperty(globalThis, 'document', { configurable: true, value: {
    createElement: () => {
      let draw
      const canvas = {
        getContext: () => ({
          clearRect: () => {},
          drawImage: (...args) => { draw = args.slice(1); calls.draws.push(draw) },
          getImageData: (x, y, readWidth, readHeight) => {
            // Crops at native resolution expose original pixels; scaled output is independent.
            const native = draw.length === 8 && draw[2] === draw[6] && draw[3] === draw[7]
            calls.reads.push({ native, width: canvas.width, height: canvas.height, readWidth, readHeight })
            if (!native) return { data: new Uint8ClampedArray(pixels) }
            const data = new Uint8ClampedArray(readWidth * readHeight * 4)
            for (let i = 0; i < data.length; i += 4) data.set(pixels, i)
            for (const pixel of originalPixels) {
              const localX = pixel.x - draw[0] + draw[4] - x
              const localY = pixel.y - draw[1] + draw[5] - y
              if (localX >= 0 && localX < readWidth && localY >= 0 && localY < readHeight)
                data[(localY * readWidth + localX) * 4 + 3] = pixel.alpha
            }
            return { data }
          },
          putImageData: value => { calls.outputPixels = [...value.data] },
        }),
        toDataURL: () => { calls.encodes++; return 'data:image/png;base64,cG5n' },
      }
      return canvas
    },
  } })
  try { await run(calls) } finally {
    URL.createObjectURL = create
    URL.revokeObjectURL = revoke
    for (const key of ['Image', 'document']) {
      if (original[key]) Object.defineProperty(globalThis, key, original[key])
      else delete globalThis[key]
    }
  }
}

const uploadFile = { name: 'image.webp', type: 'image/webp', size: 100 }

test('normalizes source and matching mask identically and converts mask luminance', async () => {
  const { prepareImageUpload } = await uploadApi()
  await withUploadBrowser({ pixels: [255, 0, 0, 255] }, async calls => {
    const source = await prepareImageUpload(uploadFile)
    const mask = await prepareImageUpload(uploadFile, source)
    assert.equal(source.dataUrl, 'data:image/png;base64,cG5n')
    assert.equal(source.originalWidth, 100)
    assert.equal(source.originalHeight, 70)
    assert.deepEqual([source.width, source.height], [96, 64])
    assert.deepEqual([mask.width, mask.height], [96, 64])
    assert.deepEqual(calls.draws.filter(draw => draw.length === 4), [[0, 0, 96, 64], [0, 0, 96, 64]])
    assert.deepEqual(calls.outputPixels, [54, 54, 54, 255])
    assert.deepEqual(calls.revoked, ['blob:local-upload', 'blob:local-upload'])
  })
})

test('rejects mismatched original mask dimensions and releases its object URL', async () => {
  const { prepareImageUpload } = await uploadApi()
  await withUploadBrowser({}, async calls => {
    await assert.rejects(prepareImageUpload(uploadFile, { originalWidth: 96, originalHeight: 64, width: 96, height: 64 }), /original image: 96 × 64/)
    assert.deepEqual(calls.draws, [])
    assert.deepEqual(calls.revoked, ['blob:local-upload'])
  })
})

test('rejects mask transparency without imposing mask rules on source alpha', async () => {
  const { prepareImageUpload } = await uploadApi()
  await withUploadBrowser({ pixels: [255, 255, 255, 0] }, async calls => {
    const source = await prepareImageUpload(uploadFile)
    assert.equal(calls.outputPixels, null)
    await assert.rejects(prepareImageUpload(uploadFile, source), /opaque mask/)
    assert.equal(calls.revoked.length, 2)
  })
})

test('releases invalid image uploads and gives a recoverable decoding error', async () => {
  const { prepareImageUpload } = await uploadApi()
  await withUploadBrowser({ decodeFails: true }, async calls => {
    await assert.rejects(prepareImageUpload(uploadFile), /could not be read/)
    assert.deepEqual(calls.revoked, ['blob:local-upload'])
  })
})

test('rejects original mask nonopacity even when resizing produces opaque pixels', async t => {
  const { prepareImageUpload } = await uploadApi()
  for (const [width, height, x, y, alpha] of [
    [8192, 32, 0, 0, 254],
    [8193, 513, 512, 511, 128],
    [8193, 513, 8192, 512, 0],
  ]) {
    await t.test(`alpha ${alpha} at (${x}, ${y}) in ${width} × ${height}`, async () => {
      await withUploadBrowser({ width, height, originalPixels: [{ x, y, alpha }] }, async calls => {
        const source = await prepareImageUpload(uploadFile)
        assert.deepEqual([source.width, source.height], [3072, height === 32 ? 32 : 192])
        await assert.rejects(prepareImageUpload(uploadFile, source), /opaque mask.*Transparent masks are not supported/)
        assert.equal(calls.encodes, 1)
        assert.equal(calls.outputPixels, null)
        assert.deepEqual(calls.revoked, ['blob:local-upload', 'blob:local-upload'])
      })
    })
  }
})

test('accepts opaque original masks with bounded inspection and the source resize', async () => {
  const { prepareImageUpload } = await uploadApi()
  for (const [width, height] of [[32, 32], [8192, 32], [8193, 513]]) {
    await withUploadBrowser({ width, height, pixels: [255, 0, 0, 255],
      originalPixels: [{ x: width - 1, y: height - 1, alpha: 255 }] }, async calls => {
      const source = await prepareImageUpload(uploadFile)
      const mask = await prepareImageUpload(uploadFile, source)
      assert.deepEqual([mask.width, mask.height], [source.width, source.height])
      assert.deepEqual([mask.originalWidth, mask.originalHeight], [width, height])
      assert.equal(mask.dataUrl, 'data:image/png;base64,cG5n')
      assert.deepEqual(calls.outputPixels, [54, 54, 54, 255])
      assert.deepEqual(calls.draws.filter(draw => draw.length === 4),
        [[0, 0, source.width, source.height], [0, 0, source.width, source.height]])
      const inspections = calls.reads.filter(read => read.native)
      assert.ok(inspections.length > 0)
      assert.ok(inspections.every(read => read.width <= 512 && read.height <= 512
        && read.readWidth <= 512 && read.readHeight <= 512))
      assert.equal(calls.encodes, 2)
      assert.deepEqual(calls.revoked, ['blob:local-upload', 'blob:local-upload'])
    })
  }
})
