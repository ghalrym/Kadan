import { useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";
import { AlertCircle, Plus, Trash2, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { SourceLink } from "./types";

export function ErrorNotice({ message }: { message: string }) {
  return message ? (
    <div className="error-notice" role="alert">
      <AlertCircle size={18} />
      <span>{message}</span>
    </div>
  ) : null;
}
export function Loading() {
  return (
    <div className="loading" role="status" aria-label="Loading">
      <div />
      <div />
      <div />
    </div>
  );
}
export function Empty({
  children,
  action,
}: {
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <p>{children}</p>
      {action}
    </div>
  );
}
export function Dialog({
  title,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    ref.current?.showModal();
    return () => ref.current?.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className={wide ? "dialog dialog-wide" : "dialog"}
      aria-labelledby={titleId}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      <div className="dialog-heading">
        <h2 id={titleId}>{title}</h2>
        <button className="icon-button" aria-label="Close" onClick={onClose}>
          <X size={20} />
        </button>
      </div>
      {children}
    </dialog>
  );
}
export function Markdown({ text }: { text: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        skipHtml
        components={{
          a: ({ children, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer noopener">
              {children}
            </a>
          ),
          img: ({ alt, ...props }) => (
            <img {...props} alt={alt || ""} loading="lazy" />
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
export function TagsField({
  value,
  onChange,
}: {
  value: string[];
  onChange: (value: string[]) => void;
}) {
  const [text, setText] = useState(value.join(", "));
  return (
    <label>
      Tags, comma separated
      <input
        value={text}
        onChange={(event) => {
          setText(event.target.value);
          onChange([
            ...new Set(
              event.target.value
                .split(",")
                .map((tag) => tag.trim())
                .filter(Boolean),
            ),
          ]);
        }}
      />
    </label>
  );
}
export function SourcesField({
  value,
  onChange,
}: {
  value: SourceLink[];
  onChange: (value: SourceLink[]) => void;
}) {
  return (
    <div className="source-editor">
      <div className="field-heading">
        <span>Source links</span>
        <button
          type="button"
          className="small-button"
          onClick={() => onChange([...value, { label: "", url: "" }])}
        >
          <Plus size={16} />
          Add link
        </button>
      </div>
      {value.map((source, index) => (
        <div className="source-row" key={index}>
          <input
            aria-label={`Source ${index + 1} label`}
            placeholder="Label"
            value={source.label}
            onChange={(event) =>
              onChange(
                value.map((item, i) =>
                  i === index ? { ...item, label: event.target.value } : item,
                ),
              )
            }
          />
          <input
            aria-label={`Source ${index + 1} URL`}
            type="url"
            placeholder="https://"
            required
            value={source.url}
            onChange={(event) =>
              onChange(
                value.map((item, i) =>
                  i === index ? { ...item, url: event.target.value } : item,
                ),
              )
            }
          />
          <button
            type="button"
            className="icon-button"
            aria-label={`Remove source ${index + 1}`}
            onClick={() => onChange(value.filter((_, i) => i !== index))}
          >
            <Trash2 size={16} />
          </button>
        </div>
      ))}
    </div>
  );
}
export function SourceLinks({ sources }: { sources: SourceLink[] }) {
  return sources.length > 0 ? (
    <div className="link-list">
      {sources.map((source, index) => (
        <a
          key={index}
          href={source.url}
          target="_blank"
          rel="noreferrer noopener"
        >
          {source.label || source.url}
        </a>
      ))}
    </div>
  ) : null;
}
export function SelectItems({
  label,
  items,
  selected,
  onChange,
}: {
  label: string;
  items: { id: string; title: string }[];
  selected: string[];
  onChange: (ids: string[]) => void;
}) {
  const [query, setQuery] = useState("");
  return (
    <details className="selection">
      <summary>
        {label}
        <span>{selected.length}</span>
      </summary>
      <input
        aria-label={`Filter ${label.toLowerCase()}`}
        placeholder="Filter"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <div className="selection-options">
        {items
          .filter((item) =>
            item.title.toLowerCase().includes(query.toLowerCase()),
          )
          .map((item) => (
            <label className="check-label" key={item.id}>
              <input
                type="checkbox"
                checked={selected.includes(item.id)}
                onChange={(event) =>
                  onChange(
                    event.target.checked
                      ? [...selected, item.id]
                      : selected.filter((id) => id !== item.id),
                  )
                }
              />
              <span>{item.title}</span>
            </label>
          ))}
        {items.length === 0 && <p className="muted">No items available.</p>}
      </div>
    </details>
  );
}
