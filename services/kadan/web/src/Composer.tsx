import { useLayoutEffect, useRef, type ReactNode } from "react";
import { ArrowUp, ArrowUpRight, ChevronDown, NotebookPen, Square, UserRound } from "lucide-react";
import type { Resource, Status } from "./types";

export function readiness(status: Status | null) {
  const storage = status?.tsuchi.state === "ready";
  return { storage, chat: storage && status?.hikari.state === "ready" };
}

function attention(status: Status | null) {
  if (!status) return null;
  if (status.tsuchi.state !== "ready")
    return { label: "Storage offline", detail: status.tsuchi.message, href: status.tsuchi.browser_url };
  const hikari = status.hikari;
  if (hikari.state === "ready") return null;
  return {
    label:
      hikari.state === "tier_required"
        ? "Choose a tier · notes only"
        : hikari.state === "model_required"
          ? "Download a model · notes only"
          : "Hikari offline · notes only",
    detail: hikari.message,
    href: hikari.settings_url,
  };
}

interface Props {
  agentName: string;
  value: string;
  onChange: (value: string) => void;
  onSend: (generate: boolean) => void;
  onStop?: () => void;
  sending: boolean;
  stopping?: boolean;
  blocked: boolean;
  status: Status | null;
  personas: Resource[];
  persona: string;
  onPersona: (id: string) => void;
  pet: ReactNode;
  autoFocus?: boolean;
}

export default function Composer({
  agentName,
  value,
  onChange,
  onSend,
  onStop,
  sending,
  stopping,
  blocked,
  status,
  personas,
  persona,
  onPersona,
  pet,
  autoFocus,
}: Props) {
  const field = useRef<HTMLTextAreaElement>(null);
  const ready = readiness(status);
  const issue = attention(status);
  const empty = !value.trim();
  const locked = sending || blocked;
  useLayoutEffect(() => {
    const node = field.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 240)}px`;
  }, [value]);

  return (
    <div className="composer-wrap">
      {issue && (
        <a
          className="attention"
          href={issue.href ?? undefined}
          aria-disabled={!issue.href}
          target="_blank"
          rel="noreferrer"
          title={issue.detail}
        >
          <span className="dot warn" />
          {issue.label}
          {issue.href && <ArrowUpRight size={13} />}
        </a>
      )}
      <form
        className={`composer ${sending ? "is-sending" : ""}`}
        onSubmit={(event) => {
          event.preventDefault();
          onSend(ready.chat);
        }}
      >
        <span className="perch">{pet}</span>
        <label className="sr-only" htmlFor="message">
          Message
        </label>
        <textarea
          id="message"
          ref={field}
          autoFocus={autoFocus}
          placeholder={ready.chat ? `Message ${agentName}` : "Write a note"}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          maxLength={16000}
          rows={1}
          // Keep typing while the agent replies, as in a text conversation.
          disabled={blocked || !ready.storage}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              onSend(ready.chat);
            }
          }}
        />
        <div className="composer-bar">
          {ready.chat && personas.length > 0 ? (
            <label className={`chip persona ${persona ? "active" : ""}`}>
              <UserRound size={14} />
              <span className="sr-only">Persona for next response</span>
              <select
                value={persona}
                onChange={(event) => onPersona(event.target.value)}
                disabled={locked}
              >
                <option value="">No persona</option>
                {personas.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.title}
                  </option>
                ))}
              </select>
              <ChevronDown size={13} />
            </label>
          ) : (
            <span />
          )}
          <div className="send-actions">
            {ready.chat && (
              <button
                className="icon note"
                type="button"
                aria-label="Save as note without a reply"
                title="Save as note · no reply"
                onClick={() => onSend(false)}
                disabled={locked || empty}
              >
                <NotebookPen size={16} />
              </button>
            )}
            {onStop && (
              <button
                className="send stop"
                type="button"
                aria-label="Stop response"
                title="Stop"
                onClick={onStop}
                disabled={stopping}
              >
                <Square size={13} fill="currentColor" />
              </button>
            )}
            <button
              className="send"
              disabled={locked || empty || !ready.storage}
              aria-label={ready.chat ? "Send message" : "Save note"}
              title={ready.chat ? "Send · Enter" : "Save note · Enter"}
            >
              {ready.chat ? <ArrowUp size={18} /> : <NotebookPen size={16} />}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
