// No additional test dependencies: compile the SDK adapter into an isolated temp
// directory, then exercise its real HTTP request path with controlled fetch.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { after, test } from 'node:test'

const output = mkdtempSync(join(tmpdir(), 'kadan-image-tests-'))
after(() => rmSync(output, { recursive: true, force: true }))
execFileSync(
  process.execPath,
  [
    resolve('node_modules/typescript/bin/tsc'),
    '--ignoreConfig',
    'src/api/images.ts',
    '--outDir',
    output,
    '--module',
    'commonjs',
    '--target',
    'es2023',
    '--lib',
    'es2023,dom',
    '--skipLibCheck',
    '--esModuleInterop',
  ],
  { stdio: 'inherit' },
)
writeFileSync(join(output, 'package.json'), '{"type":"commonjs"}')
const require = createRequire(import.meta.url)
const { requestImages, imageHistory } = require(join(output, 'images.js'))
const { client } = require(join(output, 'generated/client.gen.js'))
const signal = () => new AbortController().signal
function mockFetch(handler) {
  client.setConfig({ baseUrl: 'http://kadan.test', fetch: handler })
}

test('generate and edit send controls to typed endpoints and expose provider failure', async () => {
  for (const source of [undefined, { image: 'source.png', strength: 0.25 }]) {
    mockFetch(async (request) => {
      assert.equal(
        request.url,
        'http://kadan.test/v1/images/' + (source ? 'edits' : 'generations'),
      )
      assert.deepEqual(await request.json(), {
        prompt: 'Tree',
        count: 2,
        aspect: '4:3',
        seed: 7,
        ...source,
      })
      return Response.json({ detail: 'No image provider' }, { status: 503 })
    })
    await assert.rejects(
      requestImages(
        { prompt: 'Tree', count: 2, aspect: '4:3', seed: 7 },
        signal(),
        source,
      ),
      /No image provider/,
    )
  }
})
test('metadata cannot masquerade as a generated image', async () => {
  mockFetch(async () => Response.json({ image: { id: 'fake' } }))
  await assert.rejects(
    requestImages({ prompt: 'Tree' }, signal()),
    /without image files/,
  )
})
test('history uses backend and preserves empty history', async () => {
  mockFetch(async (request) => {
    assert.equal(request.url, 'http://kadan.test/v1/images')
    return Response.json({ images: [] })
  })
  assert.deepEqual(await imageHistory(signal()), [])
  mockFetch(async () => Response.json({}, { status: 500 }))
  await assert.rejects(imageHistory(signal()), /Could not load/)
})
test('request cancellation reaches fetch without a success fallback', async () => {
  const controller = new AbortController()
  mockFetch(async (request) => {
    controller.abort()
    assert.equal(request.signal.aborted, true)
    throw new DOMException('Cancelled', 'AbortError')
  })
  await assert.rejects(requestImages({ prompt: 'Tree' }, controller.signal))
})

test('successful generation returns deliverable URLs', async () => {
  const image = { id: 'completed', urls: ['/v1/images/completed/files/0'], seeds: [42] }
  mockFetch(async () => Response.json({ image }))
  assert.deepEqual(await requestImages({ prompt: 'Tree', count: 1 }, signal()), image)
})
