import { useEffect, useRef, useState } from 'react'
import { Panel, SectionHeading } from './Controls'

type RuntimeStatus = {
  state: 'unloaded' | 'loading' | 'ready' | 'error'
  model_id: string | null
  error: string | null
}

async function runtimeRequest(path: string, signal: AbortSignal, method = 'GET') {
  const response = await fetch(`/v1/runtime${path}`, { method, signal: AbortSignal.any([signal, AbortSignal.timeout(45000)]) })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : `Runtime request failed (${response.status}).`)
  }
  return data as RuntimeStatus
}

export default function RuntimePanel({ selectedModelId }: { selectedModelId: string | null }) {
  const [status, setStatus] = useState<RuntimeStatus | null>(null)
  const [error, setError] = useState('')
  const [pollError, setPollError] = useState('')
  const [pending, setPending] = useState(false)
  const action = useRef<AbortController | null>(null)
  const revision = useRef(0)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      const before = revision.current
      try {
        const next = await runtimeRequest('', controller.signal)
        if (!controller.signal.aborted && before === revision.current && !action.current) {
          setStatus(next)
          setPollError('')
        }
      } catch (reason) {
        if (!controller.signal.aborted && before === revision.current && !action.current) {
          setStatus(null)
          setPollError(reason instanceof Error ? reason.message : 'Cannot reach model runtime.')
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 2000)
      }
    }
    void poll()
    return () => {
      controller.abort()
      action.current?.abort()
      clearTimeout(timer)
    }
  }, [])

  async function change(path: '/load' | '/unload') {
    if (action.current) return
    const controller = new AbortController()
    action.current = controller
    revision.current += 1
    setPending(true)
    setError('')
    try {
      const next = await runtimeRequest(path, controller.signal, 'POST')
      if (!controller.signal.aborted) {
        setStatus(next)
        setPollError('')
      }
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Runtime request failed.')
    } finally {
      if (!controller.signal.aborted) setPending(false)
      if (action.current === controller) action.current = null
    }
  }

  return (
    <Panel className="stack runtime-panel">
      <SectionHeading>Chat runtime</SectionHeading>
      <p role="status">{status ? `${status.state}${status.model_id ? ` · ${status.model_id}` : ''}` : 'Runtime status unavailable'}</p>
      <p className="muted">Load the selected download to use chat. To switch models, unload first, then select and load another download.</p>
      {(error || pollError || status?.error) && <p role="alert">{error || pollError || status?.error}</p>}
      <div className="row wrap">
        <button type="button" className="button button--primary" disabled={pending || !status || !selectedModelId || status.state === 'loading' || status.state === 'ready'} onClick={() => void change('/load')}>Load selected model</button>
        <button type="button" className="button button--secondary" disabled={pending || !status || status.state === 'unloaded'} onClick={() => void change('/unload')}>{status?.state === 'loading' ? 'Cancel loading' : 'Unload model'}</button>
      </div>
    </Panel>
  )
}
