import { useEffect, useState } from "react";
import { ArrowLeft, ChevronLeft, ChevronRight, Download, Pencil } from "lucide-react";
import { agentPath, api } from "./api";
import Markdown from "./Markdown";
import { childrenOf, folderTrail, formatSize, itemDate, TYPE_LABEL } from "./spaceItems";
import { ItemIcon, contentUrl } from "./SpaceIcon";
import type { SpaceItem, SpacePageContent, SpaceSlides } from "./types";

interface Props {
  agentKey: string;
  item: SpaceItem;
  items: SpaceItem[];
  startEditing: boolean;
  onBack: () => void;
  onOpen: (item: SpaceItem) => void;
  onSaved: (item: SpaceItem) => void;
}

function PageView({ agentKey, item, items, startEditing, onOpen, onSaved }: Omit<Props, "onBack">) {
  const path = `${agentPath(agentKey)}/space/pages/${encodeURIComponent(item.id)}`;
  const [page, setPage] = useState<SpacePageContent | null>(null);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState(startEditing);
  const [title, setTitle] = useState("");
  const [markdown, setMarkdown] = useState("");
  const [saving, setSaving] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let current = true;
    setError("");
    api<SpacePageContent>(path)
      .then((value) => {
        if (!current) return;
        setPage(value);
        setTitle(value.title);
        setMarkdown(value.markdown);
      })
      .catch((e) => { if (current) setError((e as Error).message); });
    return () => { current = false; };
  }, [path, attempt]);
  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (!page || saving || !title.trim()) return;
    setSaving(true);
    setError("");
    try {
      const saved = await api<SpacePageContent>(path, { title: title.trim(), markdown, expected_version: page.version }, "PUT");
      setPage(saved);
      setEditing(false);
      onSaved(saved);
    } catch (e) {
      setError(`${(e as Error).message} Your changes are still here.`);
    } finally {
      setSaving(false);
    }
  }
  const subpages = childrenOf(items, item.id);
  if (!page) {
    return error ? (
      <div className="error feedback" role="alert">{error}<button onClick={() => setAttempt((n) => n + 1)}>Retry</button></div>
    ) : <p className="page-empty" role="status">Opening page…</p>;
  }
  if (editing) {
    const dirty = title !== page.title || markdown !== page.markdown;
    return (
      <form className="page-editor" onSubmit={save}>
        <input className="page-editor-title" aria-label="Page title" value={title} maxLength={500} autoFocus={startEditing}
          onFocus={(event) => { if (startEditing && page.title === "Untitled page") event.currentTarget.select(); }}
          onChange={(event) => setTitle(event.target.value)} />
        <textarea aria-label="Page text" value={markdown} placeholder="Start writing… Markdown works here."
          onChange={(event) => setMarkdown(event.target.value)} />
        {error && <p className="error" role="alert">{error}</p>}
        <div className="form-actions">
          <button type="button" disabled={saving} onClick={() => {
            if (dirty && !window.confirm("Discard your changes to this page?")) return;
            setTitle(page.title);
            setMarkdown(page.markdown);
            setError("");
            setEditing(false);
          }}>Cancel</button>
          <button className="primary" disabled={saving || !title.trim() || !dirty}>{saving ? "Saving…" : "Save"}</button>
        </div>
      </form>
    );
  }
  return (
    <article className="page-view">
      <header className="page-view-head">
        <h1>{page.title}</h1>
        <button onClick={() => setEditing(true)}><Pencil size={15} />Edit</button>
      </header>
      <p className="page-view-meta">Edited {itemDate(page.updated_at)}</p>
      {page.markdown.trim() ? <Markdown>{page.markdown}</Markdown> : <p className="quiet">This page is empty. Choose Edit to start writing.</p>}
      {subpages.length > 0 && (
        <section className="page-subpages" aria-label="Inside this page">
          <h2 className="page-section-title">Inside this page</h2>
          {subpages.map((child) => (
            <button key={child.id} onClick={() => onOpen(child)}>
              <span className={`item-icon small is-${child.type}`}><ItemIcon agentKey={agentKey} item={child} size={15} /></span>
              {child.title}
            </button>
          ))}
        </section>
      )}
    </article>
  );
}

function SlidesView({ agentKey, item }: { agentKey: string; item: SpaceItem }) {
  const [deck, setDeck] = useState<SpaceSlides | null>(null);
  const [error, setError] = useState("");
  const [index, setIndex] = useState(0);
  useEffect(() => {
    let current = true;
    api<SpaceSlides>(`${agentPath(agentKey)}/space/slides/${encodeURIComponent(item.id)}`)
      .then((value) => { if (current) setDeck(value); })
      .catch((e) => { if (current) setError((e as Error).message); });
    return () => { current = false; };
  }, [agentKey, item.id]);
  useEffect(() => {
    const keys = (event: KeyboardEvent) => {
      if (!deck || (event.target as HTMLElement).closest("input, textarea")) return;
      if (event.key === "ArrowRight") setIndex((n) => Math.min(n + 1, deck.slides.length - 1));
      if (event.key === "ArrowLeft") setIndex((n) => Math.max(n - 1, 0));
    };
    document.addEventListener("keydown", keys);
    return () => document.removeEventListener("keydown", keys);
  }, [deck]);
  if (error) return <p className="error" role="alert">{error}</p>;
  if (!deck) return <p className="page-empty" role="status">Opening slides…</p>;
  if (!deck.slides.length) return <p className="page-empty">This deck has no slides yet.</p>;
  const slide = deck.slides[Math.min(index, deck.slides.length - 1)];
  return (
    <div className="slides-view">
      <h1>{deck.title}</h1>
      <div className={`slide slide-${slide.layout}`}>
        {slide.title && <h2>{slide.title}</h2>}
        <div className="slide-content">
          {slide.body && <Markdown>{slide.body}</Markdown>}
          {slide.image_asset_id && <img src={contentUrl(agentKey, slide.image_asset_id)} alt="" />}
        </div>
      </div>
      <div className="slides-controls">
        <button className="icon" aria-label="Previous slide" disabled={index === 0} onClick={() => setIndex(index - 1)}>
          <ChevronLeft size={18} />
        </button>
        <span>{index + 1} / {deck.slides.length}</span>
        <button className="icon" aria-label="Next slide" disabled={index >= deck.slides.length - 1} onClick={() => setIndex(index + 1)}>
          <ChevronRight size={18} />
        </button>
      </div>
      {slide.notes && <p className="slides-notes"><strong>Notes</strong> {slide.notes}</p>}
    </div>
  );
}

function FileView({ agentKey, item }: { agentKey: string; item: SpaceItem }) {
  return (
    <div className="file-view">
      {item.type === "image" ? (
        <a href={contentUrl(agentKey, item.id)} target="_blank" rel="noreferrer" className="file-view-image">
          <img src={contentUrl(agentKey, item.id)} alt={item.title} />
        </a>
      ) : (
        <span className="file-view-icon"><ItemIcon agentKey={agentKey} item={item} size={40} /></span>
      )}
      <h1>{item.title}</h1>
      <dl className="file-view-meta">
        {item.filename && <><dt>File name</dt><dd>{item.filename}</dd></>}
        {item.media_type && <><dt>Type</dt><dd>{item.media_type}</dd></>}
        {item.size_bytes !== null && <><dt>Size</dt><dd>{formatSize(item.size_bytes)}</dd></>}
        <dt>Added</dt><dd>{itemDate(item.created_at)}</dd>
      </dl>
      <a className="button" href={contentUrl(agentKey, item.id, true)}><Download size={15} />Download</a>
    </div>
  );
}

/** One open Space item, with a path back to where it lives. */
export default function SpaceItemView(props: Props) {
  const { agentKey, item, items, onBack, onOpen } = props;
  const parent = item.parent_id ? items.find((row) => row.id === item.parent_id) : undefined;
  const trail = parent?.type === "folder" ? folderTrail(items, parent.id) : [];
  return (
    <div className="space-inner space-item">
      <nav className="space-crumbs" aria-label="Item location">
        <button className="page-back" onClick={onBack}><ArrowLeft size={16} />Back</button>
        {trail.map((crumb) => (
          <span key={crumb.id}><ChevronRight size={13} aria-hidden="true" />
            <button onClick={() => onOpen(crumb)}>{crumb.title}</button></span>
        ))}
        {parent?.type === "page" && (
          <span><ChevronRight size={13} aria-hidden="true" /><button onClick={() => onOpen(parent)}>{parent.title}</button></span>
        )}
        <span className="space-crumb-type">{TYPE_LABEL[item.type]}</span>
      </nav>
      {item.type === "page" ? <PageView {...props} />
        : item.type === "slides" ? <SlidesView agentKey={agentKey} item={item} />
          : <FileView agentKey={agentKey} item={item} />}
    </div>
  );
}
