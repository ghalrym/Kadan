import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
const source = stripTypeScriptTypes(await readFile(new URL('../api/requestStatus.ts', import.meta.url), 'utf8'))
const { requestStatus } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

test('HTTP status codes map to expected outcome classes', () => {
  assert.deepEqual(requestStatus({ type: 'Image', status: 200 }), { label: 'HTTP 200', className: 'success' })
  assert.deepEqual(requestStatus({ type: 'LLM', status: 422 }), { label: 'HTTP 422', className: 'warning' })
  assert.deepEqual(requestStatus({ type: 'TTS', status: 503 }), { label: 'HTTP 503', className: 'error' })
})
