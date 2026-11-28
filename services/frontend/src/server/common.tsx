import { useState, type ReactNode } from "react";
import { Check, Copy } from "lucide-react";

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="srv-page-head">
      <div>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {actions && <div className="srv-page-actions">{actions}</div>}
    </header>
  );
}

export function StatusPill({ tone, children }: { tone: "ok" | "warn" | "busy" | "error" | "muted"; children: ReactNode }) {
  return (
    <span className={`srv-pill ${tone}`}>
      <i aria-hidden="true" />
      {children}
    </span>
  );
}

export function CopyField({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const [failed, setFailed] = useState(false);
  return (
    <div className="srv-copy">
      <code>{value}</code>
      <button type="button" className="button" aria-label={`Copy ${label}`} onClick={() => {
        navigator.clipboard.writeText(value).then(() => {
          setCopied(true);
          setFailed(false);
          setTimeout(() => setCopied(false), 1800);
        }, () => setFailed(true));
      }}>
        {copied ? <Check size={14} /> : <Copy size={14} />}
        {copied ? "Copied" : "Copy"}
      </button>
      {failed && <span className="field-hint" role="alert">Couldn’t copy. Select the address and copy it yourself.</span>}
    </div>
  );
}

export function Meter({ label, used, total, unit }: { label: string; used: number; total: number; unit: string }) {
  const share = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  return (
    <div className="srv-meter">
      <div className="srv-meter-text">
        <span>{label}</span>
        <span>{used.toFixed(1)} of {total.toFixed(0)} {unit}</span>
      </div>
      <div className="srv-meter-bar" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={total} aria-valuenow={used}>
        <span style={{ width: `${share}%` }} />
      </div>
    </div>
  );
}
