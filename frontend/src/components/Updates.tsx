import { useEffect, useRef, useState } from 'react'
import type { UpdateStatus } from '../api/generated'
import { updateRequest } from '../api/updates'
import { Panel, SectionHeading } from './Controls'

const working = new Set(['checking', 'pulling', 'draining', 'backup', 'restarting', 'verifying', 'restoring', 'rollback'])

export function Updates() {
  const [status, setStatus] = useState<UpdateStatus | null>(null)
  const [error, setError] = useState('')
  const [code, setCode] = useState('')
  const [pending, setPending] = useState(false)
  const initial = useRef<string | null>(null)
  const disconnectedAt = useRef<number | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const next = await updateRequest('', undefined, controller.signal)
        if (controller.signal.aborted) return
        if (initial.current === null) initial.current = next.current
        if (initial.current !== next.current && next.phase === 'complete') {
          window.location.reload()
          return
        }
        disconnectedAt.current = null
        setStatus(next)
        setError('')
      } catch (failure) {
        if (controller.signal.aborted) return
        disconnectedAt.current ??= Date.now()
        setError(Date.now() - disconnectedAt.current > 120000
          ? 'Kadan has not reconnected. Check the host updater status; no further update was requested.'
          : `Reconnecting to Kadan… ${failure instanceof Error ? failure.message : 'Connection unavailable'}`)
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 3000)
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [])

  async function act(action: string, body?: { code: string } | { commit: string }) {
    if (pending) return
    setPending(true)
    try {
      const next = await updateRequest(action, body)
      setStatus(next)
      setCode('')
      setError('')
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Update request failed')
    } finally { setPending(false) }
  }

  const active = status && working.has(status.phase ?? '')
  const canUpdate = status?.authorized && status.available && ['idle', 'failed', 'complete', 'rolled_back'].includes(status.phase ?? '')
  return <section className="stack">
    <SectionHeading>Updates</SectionHeading>
    <Panel className="panel-body stack">
      <div className="model-llm-header">
        <div className="stack compact">
          <strong>Kadan {status?.current === 'development' ? 'development' : status?.current.slice(0, 12) ?? '—'}</strong>
          <p className="muted" role="status">{status?.message ?? 'Checking update availability…'}</p>
          {status?.available && <span className="mono faint">Available: {status.available.slice(0, 12)}</span>}
        </div>
        <button className={`button ${canUpdate && !active ? 'button--primary' : 'button--secondary'}`} type="button" disabled={pending || !canUpdate || !!active}
          onClick={() => { if (status?.available) void act('install', { commit: status.available }) }}>
          {active ? 'Updating…' : 'Update'}
        </button>
      </div>
      {status?.configured && !status.authorized && <form className="stack compact" onSubmit={event => { event.preventDefault(); void act('pair', { code }) }}>
        <label htmlFor="update-pair-code">Authorize updates for this browser</label>
        <p className="muted">Run <code>sudo python3 -m updater pair</code> on your server, then enter its one-time code.</p>
        <input id="update-pair-code" className="input" type="password" autoComplete="off" value={code} onChange={event => setCode(event.target.value)} />
        <button type="submit" className="button" disabled={pending || !code.trim()}>Authorize</button>
      </form>}
      {status?.authorized && status.can_cancel && <button type="button" className="button" disabled={pending} onClick={() => void act('cancel')}>Cancel update</button>}
      {(error || status?.error) && <p className="error-panel" role="alert">{error || status?.error}</p>}
    </Panel>
  </section>
}
