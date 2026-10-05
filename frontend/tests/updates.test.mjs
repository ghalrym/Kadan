import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { after, test } from 'node:test'

const output = mkdtempSync(join(tmpdir(), 'kadan-update-tests-'))
after(() => rmSync(output, { recursive: true, force: true }))
execFileSync(process.execPath, [resolve('node_modules/typescript/bin/tsc'), '--ignoreConfig',
  'src/api/updates.ts', '--outDir', output, '--module', 'commonjs', '--target', 'es2023',
  '--lib', 'es2023,dom', '--skipLibCheck'], { stdio: 'inherit' })
writeFileSync(join(output, 'package.json'), '{"type":"commonjs"}')
const { updateRequest } = createRequire(import.meta.url)(join(output, 'updates.js'))
const original = globalThis.fetch
after(() => { globalThis.fetch = original })
const status = { current: 'a'.repeat(40), phase: 'idle', configured: true }

test('automatic status is read-only; deliberate install sends exact release with CSRF header', async () => {
  const calls = []
  globalThis.fetch = async (url, options) => { calls.push({ url, options }); return Response.json(status) }
  await updateRequest()
  assert.equal(calls[0].options.method, 'GET')
  assert.equal(calls[0].options.body, undefined)
  await updateRequest('install', { commit: 'b'.repeat(40) })
  assert.equal(calls[1].options.headers['X-Kadan-Update'], '1')
  assert.equal(calls[1].options.credentials, 'same-origin')
  assert.deepEqual(JSON.parse(calls[1].options.body), { commit: 'b'.repeat(40) })
})

test('authorization, failed updates and malformed responses never become success', async () => {
  globalThis.fetch = async () => Response.json({ detail: 'Pair this browser' }, { status: 401 })
  await assert.rejects(updateRequest('install', { commit: 'b'.repeat(40) }), /Pair this browser/)
  globalThis.fetch = async () => new Response('Restarting', { status: 502 })
  await assert.rejects(updateRequest(), /HTTP 502/)
  globalThis.fetch = async () => Response.json({ ok: true })
  await assert.rejects(updateRequest(), /invalid status/)
})

test('cancellation preserves caller abort signal and cancel command carries no execution input', async () => {
  const controller = new AbortController()
  globalThis.fetch = async (url, options) => {
    controller.abort()
    assert.equal(options.signal.aborted, true)
    throw new DOMException('Aborted', 'AbortError')
  }
  await assert.rejects(updateRequest('', undefined, controller.signal), { name: 'AbortError' })
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/v1/updates/cancel')
    assert.equal(options.body, 'null')
    return Response.json(status)
  }
  await updateRequest('cancel')
})
