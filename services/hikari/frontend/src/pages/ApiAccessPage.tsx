import { useEffect, useState } from 'react'

type Schema = {
  info: { title: string; version: string; description?: string }
  paths: Record<string, Record<string, { summary?: string; description?: string }>>
}
const methods = new Set(['get', 'post', 'put', 'patch', 'delete'])

/** Show operations from the running API schema; listing is not model readiness.
 * Fetches are abortable on retry/unmount, and failures remain visible with retry.
 */
export default function ApiAccessPage() {
  const [schema, setSchema] = useState<Schema | null>(null)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    /** Validate the schema shape before rendering; ignore results after cleanup. */
    async function load() {
      try {
        const response = await fetch('/openapi.json', {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15000)]),
        })
        if (!response.ok) throw new Error(`API schema returned HTTP ${response.status}`)
        const data = await response.json() as Schema
        if (!data.info?.title || !data.paths || typeof data.paths !== 'object') {
          throw new Error('The API returned an invalid OpenAPI schema.')
        }
        for (const operations of Object.values(data.paths)) {
          if (!operations || typeof operations !== 'object' || Array.isArray(operations)) {
            throw new Error('The API returned invalid path definitions.')
          }
          for (const [method, operation] of Object.entries(operations)) {
            if (methods.has(method) && (!operation || typeof operation !== 'object'
              || (operation.summary !== undefined && typeof operation.summary !== 'string')
              || (operation.description !== undefined && typeof operation.description !== 'string'))) {
              throw new Error('The API returned an invalid operation definition.')
            }
          }
        }
        if (!controller.signal.aborted) setSchema(data)
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Cannot reach the API.')
      }
    }
    void load()
    return () => controller.abort()
  }, [refresh])

  return (
    <div className="scroll-page">
      <div className="settings-layout stack">
        <h2>API access</h2>
        <p><a href="/docs" target="_blank" rel="noreferrer">Interactive API reference</a> · <a href="/openapi.json" target="_blank" rel="noreferrer">OpenAPI JSON</a></p>
        {error && <div role="alert">{error} <button type="button" onClick={() => { setError(''); setSchema(null); setRefresh(value => value + 1) }}>Retry</button></div>}
        {!schema && !error && <p role="status">Loading API schema…</p>}
        {schema && <>
          <p>{schema.info.title} · {schema.info.version}</p>
          <p>{schema.info.description}</p>
          <table className="request-table">
            <thead><tr><th>Method</th><th>Endpoint</th><th>Operation</th></tr></thead>
            <tbody>{Object.entries(schema.paths).flatMap(([path, operations]) =>
              Object.entries(operations).filter(([method]) => methods.has(method)).map(([method, operation]) =>
                <tr key={`${method}:${path}`}><td className="mono">{method.toUpperCase()}</td><td className="mono">{path}</td><td>{operation.summary ?? operation.description ?? ''}</td></tr>,
              ),
            )}</tbody>
          </table>
        </>}
      </div>
    </div>
  )
}
