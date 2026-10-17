import { useId, type ReactNode } from 'react'
import { NavLink } from 'react-router'

// Backend-dependent controls are intentionally disabled in the static design.
export function ActionButton({
  children,
  variant = 'secondary',
  className = '',
  label,
}: {
  children: ReactNode
  variant?: 'secondary' | 'primary' | 'muted' | 'text'
  className?: string
  label?: string
}) {
  return (
    <button
      type="button"
      disabled
      aria-label={label}
      className={`button button--${variant} ${className}`}
    >
      {children}
    </button>
  )
}

export function Field({
  label,
  value = '',
  placeholder,
  multiline,
  rows = 5,
  unit,
  mono = false,
}: {
  label: string
  value?: string
  placeholder?: string
  multiline?: boolean
  rows?: number
  unit?: string
  mono?: boolean
}) {
  const id = useId()
  const className = `input${mono || unit ? ' mono' : ''}`
  return (
    <div className="field">
      <label className="eyebrow" htmlFor={id}>
        {label}
      </label>
      {multiline ? (
        <textarea
          id={id}
          className={className}
          rows={rows}
          value={value}
          readOnly
          placeholder={placeholder}
        />
      ) : (
        <div className={unit ? 'input-unit' : undefined}>
          <input
            id={id}
            className={className}
            value={value}
            readOnly
            placeholder={placeholder}
          />
          {unit && <span className="muted mono">{unit}</span>}
        </div>
      )}
    </div>
  )
}

export function SegmentedControl({
  label,
  options,
  selected,
  onChange,
  disabled = false,
}: {
  label: string
  options: string[]
  selected: string
  onChange?: (value: string) => void
  disabled?: boolean
}) {
  return (
    <div className="field">
      <span className="eyebrow">{label}</span>
      <div className="segments" role="group" aria-label={label}>
        {options.map((option) => (
          <button
            type="button"
            key={option}
            disabled={disabled || !onChange}
            onClick={() => onChange?.(option)}
            aria-pressed={option === selected}
            className={option === selected ? 'selected' : ''}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  )
}

export function ModeNavigation({
  label,
  options,
}: {
  label: string
  options: { label: string; to: string }[]
}) {
  return (
    <nav className="segments mode-navigation" aria-label={label}>
      {options.map((option) => (
        <NavLink
          key={option.to}
          to={option.to}
          end
          className={({ isActive }) => (isActive ? 'selected' : '')}
        >
          {option.label}
        </NavLink>
      ))}
    </nav>
  )
}

export function UploadPlaceholder({
  title,
  caption,
}: {
  title: string
  caption: string
}) {
  return (
    <button type="button" className="upload-placeholder" disabled>
      <span>{title}</span>
      <span className="mono faint">{caption}</span>
    </button>
  )
}

export function Panel({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return <section className={`panel ${className}`}>{children}</section>
}

export function SectionHeading({
  children,
  number,
}: {
  children: ReactNode
  number?: string
}) {
  return (
    <h2 className="section-heading">
      {number && <span className="accent mono">{number}</span>}
      {children}
    </h2>
  )
}
