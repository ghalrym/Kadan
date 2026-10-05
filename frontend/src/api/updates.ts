import type { UpdateStatus } from './generated'

/** Fixed update actions; the server independently verifies owner and release. */
export async function updateRequest(action = '', body?: { code: string } | { commit: string }, signal?: AbortSignal): Promise<UpdateStatus> {
  const response = await fetch(`/v1/updates${action ? `/${action}` : ''}`, {
    method: action ? 'POST' : 'GET',
    credentials: 'same-origin',
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000),
    headers: action ? { 'Content-Type': 'application/json', 'X-Kadan-Update': '1' } : undefined,
    body: action ? JSON.stringify(body ?? null) : undefined,
  })
  if (!response.ok) {
    let detail = `Update service returned HTTP ${response.status}`
    try { detail = (await response.json()).detail ?? detail } catch { /* Proxy restart can return HTML. */ }
    throw new Error(detail)
  }
  const value: UpdateStatus = await response.json()
  if (typeof value.current !== 'string' || typeof value.phase !== 'string' || typeof value.configured !== 'boolean') {
    throw new Error('Update service returned an invalid status')
  }
  return value
}
