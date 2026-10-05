import { useState } from 'react'
import { readFormattingPreference, saveFormattingPreference } from '../api/formatting'

/** Use the existing formatting switch; report storage failures without changing its state. */
export function FormattingSwitch() {
  const [enabled, setEnabled] = useState(readFormattingPreference)
  const [error, setError] = useState('')
  function toggle() {
    try { saveFormattingPreference(!enabled); setEnabled(!enabled); setError('') }
    catch { setError('Cannot save the formatting preference.') }
  }
  return <div><button type="button" role="switch" aria-checked={enabled} aria-label="S1-mini by Superwhisper formatting" className="switch" onClick={toggle}><span /></button>{error && <span role="alert" className="error">{error}</span>}</div>
}
