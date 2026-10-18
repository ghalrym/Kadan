import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
const source = stripTypeScriptTypes(await readFile(new URL('../src/api/requestStatus.ts', import.meta.url), 'utf8'))
const { chatPendingStatus, requestStatus } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

test('chat waits for the current turn even when earlier replies exist', () => {
  assert.equal(chatPendingStatus(true, [
    { role: 'user', text: 'Earlier question' },
    { role: 'assistant', text: 'Earlier response' },
    { role: 'user', text: 'Current question' },
  ]), 'Waiting for the chat API…')
})

test('chat reports streaming after partial response and clears status on completion or cancellation', () => {
  const messages = [{ role: 'user', text: 'test' }, { role: 'assistant', text: 'Hello' }]
  assert.equal(chatPendingStatus(true, messages), 'Receiving response…')
  assert.equal(chatPendingStatus(false, messages), null)
  assert.equal(chatPendingStatus(false, messages.slice(0, 1)), null)
  assert.equal(chatPendingStatus(true, messages.slice(0, 1)), 'Waiting for the chat API…')
})

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
