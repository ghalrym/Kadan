import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
const source = stripTypeScriptTypes(await readFile(new URL('../src/api/requestStatus.ts', import.meta.url), 'utf8'))
const { requestStatus } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

test('HTTP acceptance never means video completion', () => {
  for (const [state, label, className] of [
    [null, 'Accepted — job status unavailable', 'muted'], ['Queued', 'Queued', 'muted'],
    ['Rendering', 'Rendering', 'muted'], ['Done', 'Completed', 'success'],
    ['Failed', 'Failed', 'error'], ['Cancelled', 'Cancelled', 'warning']]) {
    assert.deepEqual(requestStatus({ type: 'Video', status: 202, video_status: state }), { label, className })
  }
})

test('failed submissions and synchronous responses retain HTTP outcome', () => {
  assert.deepEqual(requestStatus({ type: 'Video', status: 503 }), { label: 'HTTP 503', className: 'error' })
  assert.deepEqual(requestStatus({ type: 'Image', status: 200 }), { label: 'HTTP 200', className: 'success' })
})
