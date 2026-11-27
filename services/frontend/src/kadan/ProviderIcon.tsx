/** Small marks for the coding assistants shown in Work's sidebar and model picker. */
interface Props {
  harness: string;
  size?: number;
  className?: string;
  title?: string;
}

function Inkling({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      <rect x="2" y="2" width="5.4" height="5.4" rx="1.4" fill="var(--accent)" />
      <rect x="8.6" y="2" width="5.4" height="5.4" rx="1.4" fill="var(--accent)" opacity="0.45" />
      <rect x="2" y="8.6" width="5.4" height="5.4" rx="1.4" fill="var(--accent)" opacity="0.45" />
      <rect x="8.6" y="8.6" width="5.4" height="5.4" rx="1.4" fill="var(--accent)" />
    </svg>
  );
}

function Claude({ size }: { size: number }) {
  const rays = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330];
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      <g stroke="#d97757" strokeWidth="1.7" strokeLinecap="round">
        {rays.map((deg, i) => (
          <line
            key={deg}
            x1="8"
            y1="8"
            x2="8"
            y2={i % 2 === 0 ? 1.2 : 2.6}
            transform={`rotate(${deg} 8 8)`}
          />
        ))}
      </g>
    </svg>
  );
}

function Codex({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="1.25">
        {[0, 60, 120].map((deg) => (
          <ellipse key={deg} cx="8" cy="8" rx="6.2" ry="2.6" transform={`rotate(${deg} 8 8)`} />
        ))}
      </g>
    </svg>
  );
}

function Antigravity({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      <defs>
        <linearGradient id="kadan-antigravity" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#4f8cff" />
          <stop offset="0.5" stopColor="#3ccf91" />
          <stop offset="1" stopColor="#ff6b4a" />
        </linearGradient>
      </defs>
      <path
        d="M2 14 C4.5 9 5.8 2 8 2 C10.2 2 11.5 9 14 14"
        fill="none"
        stroke="url(#kadan-antigravity)"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function Grok({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
        <path d="M12.6 5.2A5.5 5.5 0 1 0 13.4 9" />
        <path d="M5 13.6 13.8 2.4" />
      </g>
    </svg>
  );
}

export default function ProviderIcon({ harness, size = 16, className, title }: Props) {
  const id = harness.toLowerCase();
  const mark =
    id === "inkling" ? (
      <Inkling size={size} />
    ) : id === "claude" ? (
      <Claude size={size} />
    ) : id === "codex" ? (
      <Codex size={size} />
    ) : id === "antigravity" ? (
      <Antigravity size={size} />
    ) : id === "grok" ? (
      <Grok size={size} />
    ) : (
      <span className="provider-icon-fallback" style={{ fontSize: Math.round(size * 0.55) }}>
        {harness.slice(0, 2).toUpperCase()}
      </span>
    );
  return (
    <span
      className={`provider-icon ${className ?? ""}`}
      style={{ width: size, height: size }}
      title={title}
      aria-hidden={title ? undefined : true}
    >
      {mark}
    </span>
  );
}
