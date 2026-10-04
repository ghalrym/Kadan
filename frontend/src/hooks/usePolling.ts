import { useEffect, useState } from 'react'

export async function readJson<T>(
  url: string,
  signal: AbortSignal,
): Promise<T> {
  const response = await fetch(url, { signal, cache: 'no-store' })
  if (!response.ok) {
    if (response.status === 404)
      throw new Error(
        'Record not found. It may have expired or the API restarted.',
      )
    throw new Error(`Monitoring API returned HTTP ${response.status}.`)
  }
  return response.json() as Promise<T>
}

/** Serialized polling: no overlap; changes/unmount abort and reject late results. */
export function usePolling<T>(url: string, interval = 3000) {
  const [snapshot, setSnapshot] = useState<{
    url: string
    data: T | null
    error: string
    updated: string | null
  }>({
    url: '',
    data: null,
    error: '',
    updated: null,
  })
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const data = await readJson<T>(
          url,
          AbortSignal.any([controller.signal, AbortSignal.timeout(10_000)]),
        )
        if (!controller.signal.aborted)
          setSnapshot({
            url,
            data,
            error: '',
            updated: new Date().toLocaleTimeString(),
          })
      } catch (reason) {
        if (!controller.signal.aborted)
          setSnapshot({
            url,
            data: null,
            error:
              reason instanceof Error
                ? reason.message
                : 'Cannot reach monitoring API.',
            updated: null,
          })
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, interval)
      }
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [url, interval, retry])
  return {
    data: snapshot.url === url ? snapshot.data : null,
    error: snapshot.url === url ? snapshot.error : '',
    updated: snapshot.url === url ? snapshot.updated : null,
    refresh: () => setRetry((value) => value + 1),
  }
}
