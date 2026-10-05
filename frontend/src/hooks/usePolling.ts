import { useEffect, useState } from 'react'

/** Fetch uncached JSON with caller-owned cancellation.
 * Reject HTTP failures (including expired-record 404s) and malformed JSON; the
 * generic type describes the contract but does not validate the response schema.
 */
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

/** Poll serially with a ten-second request timeout and a delay after completion.
 * Returns data/error/last-success time for the current URL and a manual refresh
 * callback. Failed polls clear data and its timestamp; URL changes hide the previous snapshot.
 * Cleanup aborts in-flight work and prevents late results or future polling.
 */
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
    /** Publish one current response or error, then schedule the next poll. */
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
