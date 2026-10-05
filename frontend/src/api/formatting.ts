const key = 'kadan.s1Formatting'

/** Keep this browser's transcript formatting preference across navigation. */
export function readFormattingPreference(): boolean {
  try { return localStorage.getItem(key) !== 'false' } catch { return true }
}

export function saveFormattingPreference(enabled: boolean): void {
  localStorage.setItem(key, String(enabled))
}
