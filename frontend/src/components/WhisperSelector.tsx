import { useEffect, useRef, useState } from 'react'

/** Persist the native Whisper choice without loading or downloading weights. */
export function WhisperSelector() {
  const [catalog, setCatalog] = useState<{ models: string[]; selected: string | null } | null>(null)
  const [error, setError] = useState('')
  const [pending, setPending] = useState(false)
  const inFlight = useRef(false)
  useEffect(() => {
    const controller = new AbortController()
    fetch('/v1/audio/transcriptions/models', { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error('Cannot load Whisper models.')
      setCatalog(await response.json())
    }).catch((reason: Error) => { if (!controller.signal.aborted) setError(reason.message) })
    return () => controller.abort()
  }, [])
  async function select(model: string) {
    if (inFlight.current) return
    inFlight.current = true
    setPending(true)
    setError('')
    try {
      const response = await fetch('/v1/audio/transcriptions/models', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }), signal: AbortSignal.timeout(15000) })
      if (!response.ok) throw new Error('Cannot save Whisper selection.')
      setCatalog(await response.json())
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Cannot save Whisper selection.') }
    finally { inFlight.current = false; setPending(false) }
  }
  return <div><select id="model-STT" className="input" disabled={!catalog?.models.length || pending} value={catalog?.selected ?? ''} onChange={event => void select(event.target.value)}>
    {!catalog?.models.length && <option value="">{catalog ? "No Whisper checkpoint enabled" : "Loading…"}</option>}
    {catalog?.models.map(name => <option key={name} value={name}>Whisper {name}</option>)}
  </select>{error && <p role="alert" className="error">{error}</p>}</div>
}
