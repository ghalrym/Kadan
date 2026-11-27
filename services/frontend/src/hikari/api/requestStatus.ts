import type { RequestRecord } from './generated/types.gen'

export function requestStatus(request: RequestRecord) {
  return {
    label: `HTTP ${request.status}`,
    className: request.status >= 500 ? 'error' : request.status >= 400 ? 'warning' : 'success',
  }
}
