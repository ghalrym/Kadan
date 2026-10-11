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
    [{ response: { ok: false, status: 503 } }, /No image provider/],
    [{ response: { ok: false, status: 422 } }, /HTTP 422/]]) {
    const { api } = await imageApi(response)
    await assert.rejects(api.requestImages({ prompt: 'mug', count: 1 }, new AbortController().signal), message)
  }
})
