// No additional test dependencies: compile the SDK adapter into an isolated temp
// directory, then exercise its real HTTP request path with controlled fetch.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { after, test } from 'node:test'

const output = mkdtempSync(join(tmpdir(), 'kadan-video-tests-'))
after(() => rmSync(output, { recursive: true, force: true }))
execFileSync(
  process.execPath,
  [
    resolve('node_modules/typescript/bin/tsc'),
    '--ignoreConfig',
    'src/api/video.ts',
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
const { submitVideo, loadVideos, refreshVideo, isPendingVideo } = require(join(output, 'video.js'))
const { client } = require(join(output, 'generated/client.gen.js'))
const signal = () => new AbortController().signal
const body = { prompt: ' A mountain ', negative_prompt: 'blur', duration: 8, fps: 24, resolution: '720p', aspect: '16:9' }
function mockFetch(fetch) { client.setConfig({ baseUrl: 'http://kadan.test', fetch }) }

test('real typed POST preserves editable settings and reports unavailable provider', async () => {
  mockFetch(async request => {
    assert.equal(request.url, 'http://kadan.test/v1/videos/generations')
    assert.equal(request.method, 'POST')
    assert.deepEqual(await request.json(), { ...body, prompt: 'A mountain' })
    return Response.json({ detail: 'Unavailable' }, { status: 503 })
  })
  await assert.rejects(submitVideo(body, signal()), /No job was queued/)
})

test('empty history stays empty and malformed success is rejected', async () => {
  mockFetch(async () => Response.json({ jobs: [] }))
  assert.deepEqual(await loadVideos(signal()), [])
  mockFetch(async () => Response.json({}))
  await assert.rejects(submitVideo(body, signal()), /invalid video job/)
})

test('validation rejects blank prompt before network', async () => {
  mockFetch(async () => { throw new Error('should not fetch') })
  await assert.rejects(submitVideo({ ...body, prompt: '  ' }, signal()), /Enter a prompt/)
})

test('poll only actual queued/rendering jobs, and 404 is actionable', async () => {
  assert.equal(isPendingVideo({ status: 'Done' }), false)
  assert.equal(isPendingVideo({ status: 'Queued' }), true)
  assert.equal(isPendingVideo({ status: 'Rendering' }), true)
  mockFetch(async request => {
    assert.equal(request.url, 'http://kadan.test/v1/videos/returned-id')
    return Response.json({ detail: 'Not found' }, { status: 404 })
  })
  await assert.rejects(refreshVideo('returned-id', signal()), /does not exist/)
})

test('browser cancellation propagates AbortSignal to fetch', async () => {
  let started
  const ready = new Promise(resolve => { started = resolve })
  mockFetch(request => new Promise((resolve, reject) => {
    started()
    request.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
  }))
  const controller = new AbortController()
  const pending = submitVideo(body, controller.signal)
  await ready
  controller.abort()
  await assert.rejects(pending)
})
